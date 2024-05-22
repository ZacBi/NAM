import os
import tempfile
from share_btn import community_icon_html, loading_icon_html, share_js, save_js
import huggingface_hub
import numpy as np
import utils
import models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import nethook
import torch
from models import GILLModel_1
from transformers import AutoTokenizer
# Load the model correctly
path = '/data/ruip/eva02/gill_done/checkpoints'
model = models.load_gill(path)
print("models load complished\n") 

def generate_for_prompt(input_text,  model_inputs, ret_scale_factor, num_words, temperature):
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    input_prompt = 'Q: ' + input_text + '\nA:'
    model_inputs.append(input_prompt)
    # Remove empty text.
    model_inputs = [s for s in model_inputs if s != '']
    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95
    print('Running model.generate_for_images_and_texts with', model_inputs, flush=True)
    model_outputs = model.generate_for_images_and_texts(model_inputs,
                                                        num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                        temperature=temperature, max_num_rets=1,
                                                        num_inference_steps=50, generator=g_cuda)
    print("outputs:",model_outputs[0])
    code_text = model.model.tokenizer.encode(model_outputs[0])
    print("outputs_id:",code_text)
    return model_outputs
# get matrix
def get_WoutWread(model):
    Wreadout_t = model.lm.lm_head.weight.t()
    Wout_t = model.lm.model.decoder.layers[0].fc2.weight.t()  # Assuming you are using the first layer for simplicity
    proj = Wout_t.mm(Wreadout_t).unsqueeze(0)  # Add a new dimension at the beginning
    for i in range(1, 32):
        Wout_t = model.lm.model.decoder.layers[i].fc2.weight.t()
        proj_i = Wout_t.mm(Wreadout_t).unsqueeze(0)
        proj = torch.cat((proj, proj_i), dim=0)  # Concatenate along the new dimension
    return proj # (32,16384, 50272)
def find_token(input, input_text, list, model_outputs):
    code_text = model.model.tokenizer.encode(model_outputs[0])
    # positioning
    code_id = [i-1 for i, token_id in enumerate(code_text) if token_id == input]
    if len(code_id)==0:
        print("there is no target!!")
        pass
    else: 
        target_id = code_id[0]
        layers_target = list[0][target_id][-1,:]
        layers_target = layers_target.unsqueeze(0)
        for i in range(1,32):
            layers_act_i = list[i][target_id][-1,:].unsqueeze(0)
            layers_target = torch.cat((layers_target,layers_act_i), dim=0) # act[i]
        print("layers_target:",layers_target.size())
        return layers_target # (layers, 16384)
def deal_img(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)  
            input_image = Image.open(file_path).resize(
                (224, 224)).convert('RGB')
            input_image.save(file_path)
            files_list.append(input_image)  
    return files_list
# Take the larger value of the normalization results of the two methods to determine the degree of neuron activation
# Model is Gill's opt model
# layers_act is the activation value for each layer 
# token_id: The ID of the subject in the word list
def max_of_both(model, layers_act, token_id):
    project = get_WoutWread(model)
    project = project.permute(2, 0, 1)
    project_new = project[token_id]
    expanded_project = torch.cat([project_new.unsqueeze(0)] * layers_act.shape[0], dim=0)
    expanded_project = torch.nn.Parameter(expanded_project)
    if torch.cuda.is_available():
        expanded_project = expanded_project.cuda()
    else:
        print("no device is avail")
    scores = layers_act*expanded_project
    for i in range(scores.shape[0]):
        max_value = torch.max(scores[i])
        min_value = torch.min(scores[i])
        max_value_1 = torch.max(layers_act[i])
        min_value_1 = torch.min(layers_act[i])
        normalized_scores = (scores[i] - min_value) / (max_value - min_value)
        normalized_list = (layers_act[i] - min_value_1) / (max_value_1 - min_value_1)
        max_neuron_val = torch.max(normalized_list, normalized_scores)
        if i==0:
           max_neuron = max_neuron_val.unsqueeze(0)  
        else:
           max_neuron = torch.cat((max_neuron, max_neuron_val.unsqueeze(0)), dim=0)
    # get max_neuron, Neuronal fraction obtained by the MAX method
    max_neuron = torch.mean(max_neuron, dim=0)
    return max_neuron
# The activation value is obtained by setting the hook function
         
def forward_hook(n):
    def fn(_, input, output):
        act[n].append(output.detach())
    return fn
handle_act = [model.model.lm.model.decoder.layers[n].activation_fn.register_forward_hook(forward_hook(n)) for n in
    range(32)]
# model_inputs Stores images and text entered into the GILL， You'll need to organize the input yourself  
# file_list = deal_img(path) Correct processing of the image gets
# input_text: Write a paragraph for each image
# layers_act: Activation value：（num, 32, 16384)
for i in range(len(files_list)):
    model_inputs = []
    model_inputs.append(files_list[i])
    with torch.no_grad():
        act = [[] for _ in range(32)]
        model_outputs = generate_for_prompt(input_text, model_inputs, ret_scale_factor=1.3, num_words=32, temperature=0.0)
        layers_act_i = find_token(2335, input_text, act, model_outputs)
        if layers_act_i is not None:
            if i==0:
                layers_act_i = layers_act_i.unsqueeze(0)
                layers_act = layers_act_i
            else:
                layers_act_i = layers_act_i.unsqueeze(0)
                layers_act = torch.cat((layers_act, layers_act_i), dim=0) # layers_act (B, Q, T) num，         
    model_inputs.clear()
for h in handle_act:
    del h
    


