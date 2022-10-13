from archpainter import find_cfg_by_alias
from ray_proxy.remote_env import RemoteEnvManager

cfg = find_cfg_by_alias("l2r_palette") +dict(
    job_type="waifu_diffusion"
)
g =cfg.exp_graph()
rem = g[RemoteEnvManager]
env =rem.create(num_gpus=1)
#%%
prompt = "touhou hakurei_reimu 1girl solo portrait"
def setup():
    import torch
    from torch import autocast
    from diffusers import StableDiffusionPipeline

    model_id = "hakurei/waifu-diffusion"
    device = "cuda"


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16')
    pipe = pipe.to(device)

    def generate(prompt,guidance_scale=7.5):
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=guidance_scale)["sample"][0]
        return image
    return pipe,generate




pipe,txt2img = env.put(setup)()
#%%
prompt="beautiful high quality waifu portrait 1girl solo "
res = txt2img(prompt)
from matplotlib import pyplot as plt
plt.imshow(res.fetch())
plt.show()