from os import path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers_interpret import StableDiffusionPipelineDetExplainer
from gill import layers
from torch.cuda import amp


model_path = '/mnt/workspace/model'
eva_id = 'zacbi2023/eva02'
sd_id = 'AI-ModelScope/stable-diffusion-v1-5'

# prepare eva
eva_base_path = path.join(model_path, eva_id)
eva_coco_config_rpath = 'projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py'
eva_config_path = path.join(eva_base_path, eva_coco_config_rpath)

# replace with your eva02 weights path
eva_coco_weights_rpth = 'checkpoints/eva02_L_coco_seg_sys_o365.pth'
eva_weights_path = path.join(eva_base_path, eva_coco_weights_rpth)

custum_cfg = ['MODEL.RETINANET.SCORE_THRESH_TEST', 0.5,
                'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,
                'MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH', 0.5,
                'DATASETS.TEST', [],
                'MODEL.WEIGHTS', eva_weights_path]
eva_cfg = LazyConfig.load(eva_config_path)
LazyConfig.apply_overrides(
    eva_cfg, [f"{key}={value}" for key, value in zip(custum_cfg[::2], custum_cfg[1::2])])

device = 'cuda'
eva = instantiate(eva_cfg.model).to(device)
DetectionCheckpointer(eva).load(eva_weights_path)
eva.eval()


torch_dtype=torch.bfloat16
sd_pipe = AutoPipelineForText2Image.from_pretrained(
    path.join(model_path, sd_id), torch_dtype=torch_dtype).to(device)
explainer = StableDiffusionPipelineDetExplainer(pipe=sd_pipe, det_model=eva)

# shape of llm output is  (batch_size, seq_len, hidden_dim)
raw_emb = torch.load('/mnt/workspace/data/tensor/raw_emb_tensor_cat_1.pt').to(torch_dtype)
raw_emb.requires_grad_(True)
# embedding img0-imge8
gen_prefix_embs = torch.load('/mnt/workspace/data/tensor/gen_prefix_embs_tensor_cat_1.pt').to(torch_dtype)
gen_prefix_embs.requires_grad_(True)

# gill_mapper: linear + Transformer + linear
gen_text_hidden_fcs = layers.GenTextHiddenFcs()
gill_state_dict = torch.load('/mnt/workspace/github/gill/checkpoints/gill_opt/pretrained_ckpt.pth.tar')

gen_text_hidden_fcs_state_dict = {}
for key, val in gill_state_dict['state_dict'].items():
    if 'gen_text_hidden_fcs' in key:
        prefix = 'gen_text_hidden_fcs' + key.split('gen_text_hidden_fcs')[1]
        gen_text_hidden_fcs_state_dict[prefix] = val
gen_text_hidden_fcs.load_state_dict(gen_text_hidden_fcs_state_dict)
gen_text_hidden_fcs.cuda()
gen_text_hidden_fcs.to(torch_dtype)

gen_emb = gen_text_hidden_fcs.gen_text_hidden_fcs[0](raw_emb, gen_prefix_embs)

with torch.cuda.amp.autocast(dtype=torch.float16):
    output = explainer(
        prompt_embeds=gen_emb,
        num_inference_steps=50,
        target_cls_id=15,
        raw_embeds=raw_emb,
        n_last_diffusion_steps_to_consider_for_attributions=1
    )
print(output.token_attributions)
for token, arr in output.token_attributions:
    if token== '0':
        print(token)
        tensor = torch.tensor(arr, dtype=torch.float16)
        weight_pixes = tensor.unsqueeze(0)
    else:
        print("ssss:", token)
        tensor = torch.tensor(arr, dtype=torch.float16)
        weight_pixes = torch.cat((weight_pixes, tensor.unsqueeze(0)), dim=0)
print(weight_pixes.size())
torch.save(weight_pixes, "/mnt/workspace/github/SNPA/src/gill/weigth.pt")