import os

import ray
from cytoolz import valmap
from ray import ObjectRef
from tqdm import tqdm

from archpainter.experiments.tools.nrm_editor_views import show_image
from matplotlib import pyplot as plt
from archpainter.rulebook import auto, identify_image

os.environ["RAY_PROFILING"] = "1"
from archpainter.ray_design import ray_design
from ray_remote_env.remote_env import RemoteEnvManager
import torch
rem: RemoteEnvManager = ray_design.provide(RemoteEnvManager)
#%%
txt2img = rem["txt2img_env"]["txt2img"]
txt2img2 = rem["txt2img_env2"]["txt2img"]
z_decoder = txt2img2.model.decode_first_stage

#%%

def get_all_refs(x):
    match x:
        case dict():
            return valmap(get_all_refs,x)
        case list():
            return list(map(get_all_refs,tqdm(x)))
        case tuple():
            return list(map(get_all_refs,tqdm(x)))
        case ObjectRef():
            return get_all_refs(ray.get(x))
    return x

@ray.remote
def decode_z(z):
    decoded = z_decoder(z).cpu().fetch()
    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    decoded = auto("torch,float32,BCHW,RGB,0_1",decoded)
    return decoded

@ray.remote
def decode_item(item):
    img = item["img"].cpu().fetch()
    x0 = item["pred_x0"].cpu().fetch()
    img = auto("torch,float32,BCHW,RGBA,None",img)
    x0  = auto("torch,float32,BCHW,RGBA,None",x0)
    decoded = decode_z.remote(img.value)
    decoded_x0 = decode_z.remote(x0.value)
    return dict(
        i=item["i"],
        img =img,
        x0_hat=x0,
        decoded=decoded,
        decoded_x0_hat=decoded_x0
    )

@ray.remote
def plot_decoded(item):
    item = get_all_refs(item)
    fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(12,12))
    fig.suptitle(f"restoration process step={i}")
    axes[0,0].set_title("img")
    axes[0,0].imshow(item["img"].to("numpy_rgb"))
    axes[1,0].set_title("x0_hat")
    axes[1,0].imshow(item["x0_hat"].to("numpy_rgb"))
    axes[0,1].set_title("decoded")
    axes[0,1].imshow(item["decoded"].to("numpy_rgb"))
    axes[1,1].set_title("decoded_x0_hat")
    axes[1,1].imshow(item["decoded_x0_hat"].to("numpy_rgb"))
    plt.tight_layout()
    plt.show()

#%%
history = []
plots = []
for item in tqdm(txt2img.prompt_to_generator("a board painted with hello world ")):
    decoded = decode_item.remote(item)
    history.append(decoded)
    plot = plot_decoded.options(resources=dict(local_display=1)).remote(decoded)
    plots.append(plot)

#%% great that we can make a pipeline!
ray.get(plots)

#history = list(map(lambda d:valmap(unwrap_ref,d),tqdm(ray.get(history))))
#history =
#%%

plt.imshow(unwrapped[0]["decoded"].to("numpy_rgb"))
plt.show()
#%%


#%%
#%%
# each decoding process is around 1.14s/it
#%%
for i,item in enumerate(tqdm(unwrapped)):
    # okey, this is a bit too slow for plotting...


#%%
#env
imgs = txt2img.generate_samples("hello world")
#%%
print('x')
#%%

#instances = env.get_named_instances()
show_image(identify_image(imgs.fetch()))
plt.show()
#%%
show_image(auto("torch,float32,BCHW,RGBA,None",instances["img"].cpu().fetch()))
plt.show()
#%%
#ah, this works fine? oh, no. second time wont work.
#env.run(type,args=(instances["outs"],),kwargs=dict())
img,pred_x0,e_t = [x.cpu().fetch() for x in list(instances["outs"])]
#%%
pred_x0.shape

#%%
from matplotlib import pyplot as plt
plt.hist(pred_x0.numpy().flatten(),bins=128)
plt.show()
#%%
show_image(auto("torch,float32,BCHW,RGBA,None",pred_x0))
plt.show()


