import uuid

from archpainter.auto_image import AutoImage, append_imgs
from archpainter.designs import krita_api_design
from archpainter.experiments.config.root_cfg import ROOT_CFG
from archpainter.krita.apis import KritaApis
from archpainter.rulebook import auto
from archpainter.style_transfer.deep_image_blending import RemoteDeepImageBlending
from ray_proxy import RemoteInterpreterFactory

# %%
cfg = (ROOT_CFG + krita_api_design).to_cfg()
g = cfg.exp_graph()
rif = g[RemoteInterpreterFactory]
rdib: RemoteDeepImageBlending = g[RemoteDeepImageBlending]

kpy: KritaApis = g[KritaApis]
base = kpy.get_layer("starry_night")
overlay = kpy.get_layer("goat_head_overlay")

# %%
import ray


@ray.remote
def run_dib(overlay: AutoImage, base: AutoImage):
    import wandb
    mask = overlay.convert("numpy,float32,HW,A,0_1").cast("numpy,float32,HW,L,0_1")
    mask = mask.resize_in_fmt((512, 512), "image,L,L")
    overlay = overlay.resize_in_fmt((512, 512),"image,RGB,RGB")
    base = base.resize_in_fmt((512, 512), "image,RGB,RGB")

    img = rdib.run(
        overlay,
        base,
        mask
    )
    print(img)
    wandb.init(project="CA", group="deep_image_blending", job_type="deep_image_blending")
    wandb.log(dict(
        overlay=overlay.to("wandb.Image"),
        base=base.to("wandb.Image"),
        mask=mask.to("wandb.Image"),
        img=img.to("wandb.Image")
    ), step=0)
    wandb.finish()
    return img


# %%
task = run_dib.remote(overlay, base)
# %%
append_imgs(base,overlay,task).show_plot()
