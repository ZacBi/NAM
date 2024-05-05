# %% [markdown]
# # Stable Diffusion & Eva

# %%
import os
from os import path

import pyarrow as pa
import torch
from datasets import load_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from diffusers import AutoPipelineForText2Image
from huggingface_hub import hf_hub_download
from modelscope import AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks
from transformers import AutoModel, LlamaModel

# %%
# 拉取到本地
base_url = r'/mnt/workspace'
ckpt_base_pth = path.join(base_url, 'model')
sd_id = r'AI-ModelScope/stable-diffusion-2-1'
sd_path = os.path.join(ckpt_base_pth, sd_id)

# %%
device = 'gpu' if torch.cuda.is_available() else 'cpu'
# device_map = '0'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# from_pretrained_dict = {'device_map': device_map, 'torch_dtype' : torch_dtype, 'revision': 'v1.0.1'}
from_pretrained_dict = {'torch_dtype' : torch_dtype,  'variant': 'fp16'}

def prompt_tensors_to_cuda(token_tensors):
    for k, v in token_tensors.items():
        token_tensors[k] = v.to('cuda')
    return token_tensors

# %%
# sd pipeline
sd_pipeline = AutoPipelineForText2Image.from_pretrained(sd_path, **from_pretrained_dict).to('cuda')

# %%
from diffusers_interpret import StableDiffusionPipelineExplainer

explainer = StableDiffusionPipelineExplainer(sd_pipeline)
prompt = "A cat sits on the chair"
with torch.autocast('cuda'):
    output = explainer(
        prompt, 
        num_inference_steps=50,
        n_last_diffusion_steps_to_consider_for_attributions = 1
    )

# %%
output.image

# %%
output.token_attributions

# %%



