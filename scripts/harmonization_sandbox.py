from typing import Callable

from ipywidgets import VBox
from rx.subject import BehaviorSubject
from tqdm import tqdm

from archpainter.auto_image import AutoImage, append_imgs
from archpainter.designs import krita_api_design
from archpainter.experiments.config.config_aliases import find_cfg_by_alias
from archpainter.models.adain_diffusion.adain_diffusion import EpsModel, DefaultEpsModelForDiffusionWrapper, \
    AdainEpsModelForDiffusionWrapper
from archpainter.models.adain_diffusion.adain_diffusion_design import img2img_generation_design, \
    img2img_harmonization_design
from archpainter.ray_util.cluster_design import cluster_design
from pinject_design import Injected, Designed, Design
from pinject_design.di.graph import ExtendedObjectGraph
from ray_proxy import Var, IRemoteInterpreter
from ray_proxy.injected_resource import InjectedResource
from stable_diffusion.img2img import img2img_env

# %%
cfg = find_cfg_by_alias("l2r_palette") + dict(
    job_type="harmonization_sandbox"
) + krita_api_design + Design().bind_provider(
    img2img_env=img2img_env
)
g = cfg.exp_graph()
ray = g["ray"]
# env = g[img2img_env]
# env
# %%
from archpainter.krita.apis import KritaApis

kpy: KritaApis = g.provide(KritaApis)
base = kpy.get_layer("starry_night")
overlay = kpy.get_layer("goat_head_overlay")

# layers = kpy.krita_layer_getter.get_layers(["base","overlay"])
# base = layers["base"]
# overlay = layers["overlay"]
# %%
from matplotlib import pyplot as plt
from archpainter.experiments.tools.nrm_editor_views import show_image

append_imgs(base, overlay).show_plot()
# %%
from archpainter.tasks.harmonizers import StableRemoteDiffusionHarmonizer, StableDiffusionHarmonizer, \
    StableDiffusionHarmonizerV3, SimplifiedHarmonizer, overwrite_with_alpha, mask_out_alpha

# %%

remote_harmonizer = Designed.bind(StableDiffusionHarmonizerV3).override(
    img2img_generation_design.bind_instance(
        img2img_generation_design=img2img_generation_design + img2img_harmonization_design)
)


def provide_harmonizer(img2img_env: IRemoteInterpreter):
    harmonizer = StableRemoteDiffusionHarmonizer(
        img2img_env,
        remote_harmonizer,
        get_control_widget=lambda: (VBox(), BehaviorSubject(dict())))
    return harmonizer


# harmonizer_resource = Injected.bind(provide_harmonizer)
cld = cluster_design.bind_provider(
    harmonizer=InjectedResource(Injected.bind(provide_harmonizer), "ondemand", 4)
)
sch = cld.to_scheduler()
sch.set_max_issuables(
    gpu=4,
    img2img_env=4,
    # gatys_session=1
)
# %%
# this is very complicated, since we need three designs.
# 1. local design for creating local stuff
# 2. remote design for customizing img2img env
# 3. custom design for overriding the customized img2img env.(session)
# this architecture is very flexible, but is complicated since the structure becomes chaotic.

# %%

@ray.remote
def run_exp(config):
    # we can make this remote since the objects in harmonizer is serializable
    import wandb
    wandb.init(
        project="CA",
        job_type="harmonization_sandbox",
        config=config
    )
    ol = mask_out_alpha(overlay) if wandb.config["masked_overlay"] else overlay
    wandb.log(dict(
        base=base.to("wandb.Image"),
        overlay=ol.to("wandb.Image"),
        combined=overwrite_with_alpha(base, overlay).to("wandb.Image"),
    ))
    ovr = Design()
    if wandb.config["adain_src"] is not None:
        ovr = ovr.bind_instance(
            style_img=dict(base=base, overlay=ol)[wandb.config["adain_src"]]
        )
    if not wandb.config["use_adain"]:
        ovr = ovr.bind_class(
            eps_model=DefaultEpsModelForDiffusionWrapper
        )
    else:
        ovr = ovr.bind_class(
            eps_model=AdainEpsModelForDiffusionWrapper
        )
    # we, need to visualize the session's structure.
    with sch["harmonizer"] as harmonizer:
        prog = harmonizer.harmonize(
            base,
            overlay,
            overlay.convert("numpy,uint8,HWC,A,0_255").cast("numpy,uint8,HWC,L,0_255"),
            extras=dict(
                strength=wandb.config["strength"],
                prompt=wandb.config["prompt"],
                override=ovr
            )
        )
        logs = list(prog)
    for step, img in tqdm(enumerate(logs),desc="logging to wandb..."):
        wandb.log(dict(img=img.auto_first().to("wandb.Image")), step=step)
    wandb.finish()
    return img


# %%
ray.get(run_exp.remote(
    dict(
        use_adain=True,
        adain_src="overlay",
        masked_overlay=True,
        strength=0.3,
        prompt="",
    ))
)

# %%
searches = []
for adain_src in ["overlay", "base", None]:
    for masked_overlay in [True, False]:
        for strength in [0.05,0.1]:
            for prompt in [""]:
                searches.append(dict(
                    use_adain=adain_src is not None,
                    adain_src=adain_src,
                    masked_overlay=masked_overlay,
                    strength=strength,
                    prompt=prompt,
                    group="harmonization_adain_hp_check"
                ))
imgs = []
for conf in tqdm(searches):
    imgs.append(run_exp.remote(conf))
# %%
append_imgs(imgs).show_plot()
# ray.get(imgs)
# %%, hmm the UNet is too black-box to apply any modification?
# what would be the directions?

img.show_plot()
# %%
