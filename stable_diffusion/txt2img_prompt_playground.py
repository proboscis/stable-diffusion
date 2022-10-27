from pytorch_lightning import seed_everything

from archpainter.auto_image import append_imgs
from archpainter.experiments.config.root_cfg import ROOT_CFG
from archpainter.experiments.tools.nrm_editor_views import show_image
from archpainter.ray_design import ray_design
from archpainter.ray_util.cluster_design import cluster_design
from archpainter.rulebook import identify_image
from data_tree.util import ensure_path_exists
from pinject_design import Design, injected_factory

from stable_diffusion.txt2img import txt2img_env, RemoteTextToImage

#%%
g = (ROOT_CFG + Design().bind_instance(
    scheduler_design = cluster_design.bind_provider(
    ).override_issuable(
        gpu_env=4,
        txt2img_env=1,
        txt2img=1,
        img2img_env=1,
    )
).bind_provider(
    sch = lambda scheduler_design,ray:scheduler_design.to_scheduler()
)).to_cfg().exp_graph()
from matplotlib import pyplot as plt
sch = g["sch"]
#%%
with sch["txt2img":1] as txtimgs:
    for i,t in enumerate([txtimgs]):
        t.env.put(seed_everything)(i)
#%%
prompt = "beautiful concept art of a landscape in Japan"
prompt = "beautiful concept art of armored core in battlefield"
prompt = "beautiful concept art of codegeass characters"
prompt = "beautiful concept art of armored core in battlefield by Hokusai, Taken by Canon 4D"
prompt = "captured princess"
prompt = """High quality concept art of a mage standing at the center of a magic circle 
and exercising great magic. Center of the legendary chapel, many crowds. 
High fantasy. Golden. XXXXXXXX composition. XXXXX light. 
The glow of magic. XXXXXXXX XXXXXXXX. art station trending,
 octane render, 8k, by XXXXXXXX and XXXXXXXX and XXXXXXXX, golden, crazy detailing"""
prompt = """High quality concept art of an armored core in a battlefield. Sci-fi. 8k, crazy detailing"""
prompt = """High quality concept art of Robot in Mt. Fuji, crazy detailing"""
prompt = """High quality concept art of a woman wearing crystal dress in the ruined ancient chapel."""
prompt = """High quality picture of an ancient ruin. Taken by Sony alpha 7 III"""
prompt = """High quality picture of an ancient ruin. Taken by Sony alpha 7 III"""
prompt = """High quality map of the elden ring world"""
prompt = """a research scientist at cyber agent"""
prompt = """High quality concept art of an abandoned shack. Fuji, crazy detailing, painted by Monet"""
prompt = "pikachu in Ukiyoe style by Hokusai"
prompt = "a beautiful portrait illustration of a waifu Shinobi character in anime style"
prompt = "A moe kawaii face of a girl with glasses drawn by Kyoani"
prompt = "a beautiful concept art of stylish sci-fi building in dark bladerunner style"
prompt = "a sister praying in a beautiful mosk, lit by shiny sunlight"
prompt = "Shizuoka prefecture, under heavy flood"
prompt = "Shizuoka prefecture"
prompt = "Kanagawa prefecture"
prompt = "a photo of a face of a confused male scientist"
prompt = "a goat painted by Gogh"
prompt = "pile of tomatoes and marbles on the floor"
import ray
@ray.remote
def r_txt2img(**kwargs):
    with sch["txt2img"] as t:
        return t.generate_samples(**kwargs)

img = [r_txt2img.remote(prompt=prompt,n_samples=1,width=512,height=512) for i in range(1)]
append_imgs(img).show_plot()
for i,_img in enumerate(append_imgs(img).to("[image,RGB,RGB]")):
    path = f"free_images/sd/{prompt}_{i}".replace(" ","_") + ".png"
    ensure_path_exists(path)
    print(path)
    _img.save(path)
#%%
%cd /Users/kento/repos/archpainter
#%%
#%%
#%%
append_imgs(img).to("[image,RGB,RGB]")
#%%
#img = txt2img.generate_samples(prompt,n_samples=1,width=1536,height=384).fetch()
imgs = identify_image(img)
imgs.value
plt.figure(figsize=(24,12))
plt.imshow(imgs.value[0])
plt.tight_layout()
plt.show(dpi=600)

#%%
# ok now let me do some fine tuning, on pixiv dataset or danbooru dataset?

for img in imgs.to("[image,RGB,RGB]"):
    plt.figure(figsize=(12,12))
    show_image(identify_image(img))
    plt.tight_layout()
    plt.show(dpi=600)

#%%
img.to("image,RGB,RGB").show()
#%%
# now, how can I do inpainting and such?

