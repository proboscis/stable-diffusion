from archpainter.experiments.tools.nrm_editor_views import show_image
from archpainter.rulebook import auto, identify_image
from ray_jobs.ray_design import ray_design
import ray
from scripts.txt2img import Txt2ImgActor
ray_design.provide("ray")
# BEWARE: start txt2img actor by using txt2img.py.
txt2img = ray.get_actor("txt2img")
from matplotlib import pyplot as plt

#%%
# well, no need to use gradio at all now...
prompt = "a game programmer at bandai namco"
img = ray.get(txt2img.generate_grid.remote(prompt))
img = identify_image(img)
#%%
prompt = "a programmer at cyberagent"
img = ray.get(txt2img.generate_samples.remote(prompt))
imgs = identify_image(img)
#%%

for img in imgs.to("[image,RGB,RGB]"):
    plt.figure(figsize=(12,12))
    show_image(identify_image(img))
    plt.tight_layout()
    plt.show(dpi=600)

#%%
#img.to("image,RGB,RGB").show()

plt.figure(figsize=(24,24))
show_image(imgs)
plt.tight_layout()
plt.show(dpi=600)
#%%
img.to("image,RGB,RGB").show()
#%%
# now, how can I do inpainting and such?

