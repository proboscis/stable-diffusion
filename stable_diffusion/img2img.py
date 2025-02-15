"""make variations of input image"""

import argparse, os, sys, glob
from dataclasses import dataclass
from typing import Sequence, Iterator, Callable, cast

import PIL
import torch
import numpy as np
from PIL.Image import Resampling
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext, contextmanager
import time
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler,MyDDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from omni_converter import IAutoData
from archpainter.ray_design import ray_design

from pinject_design import Design, Injected
from pinject_design.di.graph import ExtendedObjectGraph
from pinject_design.di.injected import InjectedFunction
from ray_proxy import IRemoteInterpreter, RemoteInterpreterFactory, Var


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def prep_image(path_or_img):
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    else:
        img = path_or_img

    image = img.convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def parse(args: Sequence[str]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args(args)
    return opt


def main():
    import sys
    opt = parse(sys.argv[1:])

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image = prep_image(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc, )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


def serve_img2img(argv):
    print(os.getcwd())
    os.chdir("../stable-diffusion")
    opt = parse(argv)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    my_sampler = MyDDIMSampler(sampler)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    def get_enc_of_img(img: PIL.Image.Image, strength):
        init_image = prep_image(img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")
        return init_latent, t_enc

    def decode(samples):
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples

    @contextmanager
    def work_scope():
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    yield

    def img2img_gen(img, prompt, strength):
        prompts = batch_size * [prompt]
        init_latent, t_enc = get_enc_of_img(img, strength)
        with work_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            c = model.get_learned_conditioning(prompts)

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(init_latent,
                                              torch.tensor([t_enc] * batch_size).to(device))
            # decode it
            for x_dec in my_sampler.decode_generator(
                    z_enc, c, t_enc,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc, ):
                yield x_dec

    def img2img(img, prompt: str, strength):
        prompts = batch_size * [prompt]
        init_latent, t_enc = get_enc_of_img(img, strength)
        with work_scope():
            all_samples = list()
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            c = model.get_learned_conditioning(prompts)

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(init_latent,
                                              torch.tensor([t_enc] * batch_size).to(device))
            # decode it
            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                     unconditional_conditioning=uc, )
            x_samples = decode(samples)
            all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            return dict(
                grid=img,
                samples=x_samples
            )

    from archpainter.tasks.harmonizers import StableDiffusionHarmonizer
    from archpainter.models.diffusion.ddim_encoding import DdimEncodingHistoryProvider
    sampler.make_schedule(
        ddim_num_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        verbose=False
    )
    design = Design(
        classes=[
            StableDiffusionHarmonizer,
            DdimEncodingHistoryProvider
        ]
    ).bind_instance(
        model=model,
        sampler=sampler,
        get_enc_of_img=get_enc_of_img,
        precision_scope=precision_scope,
        opt=opt,
        batch_size=batch_size,
        device=device
    )
    # for some reason this graph cannot be accessed from remote?
    graph = design.to_graph()

    return {**globals(), **locals()}


def get_img2img_env(remote_interpreter_factory: RemoteInterpreterFactory, force_create, name) -> IRemoteInterpreter:
    # rem: RemoteEnvManager = ray_design.to_graph()[RemoteEnvManager]
    # rem.destroy("img2img_env")
    if force_create:
        remote_interpreter_factory.destroy(name)
        env = remote_interpreter_factory.create(name=name, num_gpus=1)
        # env = remote_interface_factory.get_or_create("img2img_env", num_gpus=1)
    else:
        env = remote_interpreter_factory.get(name)
    if not "img2img_vars" in env:
        env["img2img_vars"] = env.put(serve_img2img)([])
    return env

def get_annon_img2img_env(remote_interpreter_factory:RemoteInterpreterFactory):
    env = remote_interpreter_factory.create(num_gpus=1)
    env["img2img_vars"] = env.put(serve_img2img)([])
    print(env["img2img_vars"])
    return env



img2img_env = Injected.bind(get_img2img_env,
                            force_create=Injected.pure(True),
                            name=Injected.pure("img2img_env"))
annon_img2img_env = Injected.bind(get_annon_img2img_env)

img2img_graph:Injected[Var[ExtendedObjectGraph]] = img2img_env.proxy["img2img_vars"]["graph"].eval()

img2img_env_client = Injected.bind(get_img2img_env,
                                   force_create=Injected.pure(False),
                                   name=Injected.pure("img2img_env"))

if __name__ == "__main__":
    # okey, I wanna be able to start this actor from the other places.
    rem: RemoteInterpreterFactory = ray_design.to_graph()[RemoteInterpreterFactory]
    rem.destroy("img2img_env")
    env = rem.get_or_create("img2img_env", num_gpus=1)
    env["img2img_vars"] = env.put(serve_img2img)([])
    print(env["img2img_vars"].keys())
    while True:
        print("serving...")
        time.sleep(10000)
