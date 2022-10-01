from archpainter.experiments.config.config_aliases import find_cfg_by_alias
from designs import krita_api_design
from stable_diffusion.img2img import img2img_env

cfg =find_cfg_by_alias("l2r_palette")+dict(
    job_type="harmonization_sandbox"
) + krita_api_design
g = cfg.exp_graph()
env = g[img2img_env]
env
#%%
from archpainter.krita.apis import KritaApis
kpy:KritaApis = g.provide(KritaApis)
layers = kpy.krita_layer_getter.get_layers(["base","overlay"])
base = layers["base"]
overlay = layers["overlay"]
#%%
from matplotlib import pyplot as plt
from archpainter.experiments.tools.nrm_editor_views import show_image
show_image(base)
plt.show()
#%%
from archpainter.tasks.harmonizers import StableRemoteDiffusionHarmonizer
harmonizer = StableRemoteDiffusionHarmonizer(env)
#%%
prog = harmonizer.harmonize(
    base,
    overlay,
    overlay.convert("numpy,uint8,HWC,A,0_255").cast("numpy,uint8,HWC,L,0_255"),
    extras=dict(
        strength=0.5,
        prompt="forest and a car"
    )
)
for img in prog:
    print("update")
    plt.figure()
    show_image(img)
    plt.show()
print("done")
#%%
show_image(base)
plt.show()
#%%
0

