"""SAMPLING ONLY."""
from dataclasses import dataclass, field
from typing import Callable

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


@dataclass
class PLMSConfig:
    repeat_noise: bool = field(default=False)
    use_original_steps: bool = field(default=False)
    quantize_denoised: bool = field(default=False)
    temperature: float = field(default=1.0)
    noise_dropout: float = field(default=0.)
    score_corrector: object = field(default=None)
    unconditional_guidance_scale: float = field(default=1.)
    unconditional_conditioning: float = field(default=None)


class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.corrector_kwargs = kwargs.get("corrector_kwargs", None)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conf: PLMSConfig,
               conditioning=None,
               callback=None,
               img_callback=None,
               eta=0.,
               mask=None,
               x0=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               **kwargs
               ):
        """
        so many parameters are set, but txt2img parameters are
          S=opt.ddim_steps,
          conditioning=c,
          batch_size=opt.n_samples,
          shape=shape,
          verbose=False,
          unconditional_guidance_scale=opt.scale,
          unconditional_conditioning=uc,
          eta=opt.ddim_eta,
          x_T=start_code
        okey? S == NStep.

        :return:
        """
        # 1. yield things.
        self.assert_conditioning_shape(batch_size, conditioning)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates = self.plms_sampling(conditioning,
                                                    size,
                                                    conf,
                                                    x_T=x_T,
                                                    ddim_use_original_steps=False,
                                                    callback=callback,
                                                    mask=mask, x0=x0,
                                                    img_callback=img_callback,
                                                    log_every_t=log_every_t,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def sample_generator(self, S, batch_size, shape, conf: PLMSConfig, conditioning, eta, x_T=None):
        self.assert_conditioning_shape(batch_size, conditioning)
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
        yield from self.plms_sampling_gen(conditioning, shape, conf, img=x_T)

    def assert_conditioning_shape(self, batch_size, conditioning):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

    @torch.no_grad()
    def plms_sampling_gen(self, cond, shape, conf: PLMSConfig,
                          img,
                          ddim_use_original_steps=False,
                          timesteps=None,
                          mask=None,
                          x0=None, ):
        device = self.model.betas.device

        if img is None:
            img = torch.randn(shape, device=device)
        else:
            img = img

        b = shape[0]

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(img, cond, ts, index, conf,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            yield dict(
                img=img,
                pred_x0=pred_x0,
                e_t=e_t,
                i=i, index=index,
                total_steps=total_steps
            )
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

    @torch.no_grad()
    def plms_sampling(self, cond, shape, conf: PLMSConfig,
                      x_T=None,
                      ddim_use_original_steps=False,
                      callback=None,
                      timesteps=None,
                      mask=None,
                      x0=None,
                      img_callback=None,
                      log_every_t=100,
                      ):
        # so much configurations that makes it too hard too hard to modify this code.
        # this code is toooo complex because of many parameters which we dont know how it affects each other.
        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        for item in self.plms_sampling_gen(
                cond=cond,
                shape=shape,
                conf=conf,
                img=img,
                ddim_use_original_steps=ddim_use_original_steps,
                timesteps=timesteps,
                mask=mask, x0=x0
        ):
            i = item["i"]
            index = item["index"]
            img, pred_x0, e_t = item["img"], item["pred_x0"], item["e_t"]
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == item["total_steps"] - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, conf: PLMSConfig, t_next=None, old_eps=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if conf.unconditional_conditioning is None or conf.unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([conf.unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + conf.unconditional_guidance_scale * (e_t - e_t_uncond)

            if conf.score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = conf.score_corrector.modify_score(self.model, e_t, x, t, c, **self.corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if conf.use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if conf.use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if conf.use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if conf.use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if conf.quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, conf.repeat_noise) * conf.temperature
            if conf.noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=conf.noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
