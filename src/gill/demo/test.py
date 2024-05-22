import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import tempfile
from share_btn import community_icon_html, loading_icon_html, share_js, save_js
import huggingface_hub
import gradio as gr
import os
# from gill import utils
# from gill import models

# import evals.utils as utils
# import evals.models as models
import utils
import models
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
import nethook
# torch.cuda.set_device(6)



os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "False"

css = """
    
    #chatbot { min-height: 300px; }
    
    
    #save-btn {
        background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
    }
    #save-btn:hover {
        background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
    }
    #share-btn {
        background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
    }
    #share-btn:hover {
        background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
    }
    
    #gallery { z-index: 999999; }
    #gallery img:hover {transform: scale(2.3); z-index: 999999; position: relative; padding-right: 30%; padding-bottom: 30%;}
    #gallery button img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; padding-bottom: 0;}
    
    @media (hover: none) {
        #gallery img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; 0;}
    }
    .html2canvas-container { width: 3000px !important; height: 3000px !important; }
"""

examples = [
    'examples/car.png',
    'examples/cake.png',
    'examples/house.png',
    'examples/maple leaf.png',
    'examples/train.png',
]

# Download model from HF Hub.
# ckpt_path = huggingface_hub.hf_hub_download(
#    repo_id='jykoh/gill', filename='pretrained_ckpt.pth.tar', local_dir='/data1/ruip/gill/gill-main_1/download')
# decision_model_path = huggingface_hub.hf_hub_download(
#     repo_id='jykoh/gill', filename='decision_model.pth.tar', local_dir='/data1/ruip/gill/gill-main_1/download')
# args_path = huggingface_hub.hf_hub_download(
#    repo_id='jykoh/gill', filename='model_args.json', local_dir='/data1/ruip/gill/gill-main_1/download')
path = '/data1/ruip/gill/gill-main_1/checkpoints/gill_opt'
# model = models.load_gill('./', args_path, ckpt_path, decision_model_path)
model = models.load_gill(path)
print("models load complished\n")



def upload_image(state, image_input):
    conversation = state[0]
    chat_history = state[1]
    
    
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    
    input_image.save(image_input.name)  # Overwrite with smaller image.
    
    
    conversation += [(f'<img src="./file={image_input.name}" style="display: inline-block;">', "")]
    
    return [conversation, chat_history + [input_image, ""]], conversation


def reset():
    return [[], []], []


def reset_last(state):
    
    conversation = state[0][:-1]
    
    chat_history = state[1][:-2]
    return [conversation, chat_history], conversation


def save_image_to_local(image: Image.Image):
    # TODO(jykoh): Update so the url path is used, to prevent repeat saving.
    
    
    
    filename = next(tempfile._get_candidate_names()) + '.png'
    
    image.save(filename)
    return filename







def generate_for_prompt(input_text,  model_inputs, ret_scale_factor, num_words, temperature):
    
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    # Ignore empty inputs.
    
    #if len(input_text) == 0:
    #    return state, state[0], gr.update(visible=True)
    
    input_prompt = 'Q: ' + input_text + '\nA:'
    # conversation = state[0]
    # chat_history = state[1]
    # print('Generating for', chat_history, flush=True)
    
    
    
    # model_inputs = chat_history
    model_inputs.append(input_prompt)
    # Remove empty text.
    model_inputs = [s for s in model_inputs if s != '']
    
    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95
    # code_2 = model.model.tokenizer.encode(model_inputs[1])
    print('Running model.generate_for_images_and_texts with', model_inputs, flush=True)
    model_outputs = model.generate_for_images_and_texts(model_inputs,
                                                        num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                        temperature=temperature, max_num_rets=1,
                                                        num_inference_steps=50, generator=g_cuda)
    return model_outputs
    print('model_outputs', model_outputs, ret_scale_factor, flush=True)
    tokenized_text = model.model.tokenizer.tokenize(model_outputs[0])
    code = model.model.tokenizer.encode(model_outputs[0]) 
    print(tokenized_text)
    print(code)
    
    
    
    """
    response = ''
    
    text_outputs = []
    
    for output_i, p in enumerate(model_outputs):
        
        if type(p) == str:
            
            if output_i > 0:
                response += '<br/>'
            # Remove the image tokens for output.
            
            text_outputs.append(p.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', ''))
            
            response += p
            if len(model_outputs) > 1:
                response += '<br/>'
        elif type(p) == dict:
            # Decide whether to generate or retrieve.
            if p['decision'] is not None and p['decision'][0] == 'gen':
                image = p['gen'][0][0]#.resize((224, 224))
                filename = save_image_to_local(image)
                response += f'<img src="./file={filename}" style="display: inline-block;"><p style="font-size: 12px; color: #555; margin-top: 0;">(Generated)</p>'
            else:
                image = p['ret'][0][0]#.resize((224, 224))
                filename = save_image_to_local(image)
                response += f'<img src="./file={filename}" style="display: inline-block;"><p style="font-size: 12px; color: #555; margin-top: 0;">(Retrieved)</p>'

    chat_history = model_inputs + \
        [' '.join([s for s in model_outputs if type(s) == str]) + '\n']
    # Remove [RET] from outputs.
    conversation.append((input_text, response.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', '')))

    # Set input image to None.
    print('state', state, flush=True)
    print('updated state', [conversation, chat_history], flush=True)
    return [conversation, chat_history], conversation, gr.update(visible=True), gr.update(visible=True)
"""
def find_token(input, input_text, list, model_outputs):
    code = model.model.tokenizer.encode(input, add_special_tokens=False)[0]
    
    print("code:", code)
    layers_target = []
    code_text = model.model.tokenizer.encode(model_outputs[0])
    print("code_text: ", code_text)
    code_model_text = model.model.tokenizer.encode(model_inputs[1])
    print(code_text[2])
    code_id = [i for i, token_id in enumerate(code_text) if token_id == 2335]
    if len(code_id)==0:
        print("there is no target!!")
        pass
    else: 
        
        
        target_id = len(code_model_text) + 4 + code_id[0]
        for i in range(32):
            layers_target.append(list[i][target_id][-1,:]) # act[i]
        return layers_target
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
# def cal_prob(input):
act = [[] for _ in range(32)]
def forward_hook(n):
    def fn(_, input, output):
        act[n].append(output.detach())
    return fn
handle_act = [model.model.lm.model.decoder.layers[n].activation_fn.register_forward_hook(forward_hook(n)) for n in
              range(32)]    
    
path_1 = '/data1/ruip/gill/gill-main_2/test_dog_data'
input_text = "What is this?"

to_return = []
to_return_1 = []
model_inputs = []
model_outputs = []

layers_act = [[] for _ in range(2)]
tenth_max_act = [[] for _ in range(2)]
tenth_ave_act = []
files_list = deal_img(path_1)
for i in range(2):
    print("生成第{}个".format(i))
    model_inputs.append(files_list[i])
    with torch.no_grad():
        # with nethook.TraceDict(
        #     module=model.model.lm.model,
        #     layers=[layers[i] for i in range(32)],
        #     retain_input=False,
        #     retain_output=True,
        # ) as tr:
        #     generate_for_prompt(input_text, model_inputs, ret_scale_factor=1.0, num_words=32, temperature=0.0)
        #     model_outputs = tr.act
        #     print(tr.act[0])
        #     to_return.append(tr.output)
        model_outputs = generate_for_prompt(input_text, model_inputs, ret_scale_factor=1.0, num_words=32, temperature=0.0)
        if i < 2:
            
            layers_act[i] = find_token("dog", input_text, act, model_outputs)
        
        # for index_1, elem in enumerate(layers_act[i]):
        #     
        #     # print("elem.size", elem.size)
        #     top_values, top_indices = torch.topk(elem, k=10)
        #     tenth_max_act[i].append([(index_1, top_indices[i].item()) for i, index in enumerate(top_indices)])
        # print(tenth_max_act[i])
    # print(len(to_return))
    # print(to_return)
    # to_return_1.append(to_return)
    # print(len(model_outputs))
    # print(model_outputs[0]) 
    # for m in range(32):
    #     print(model_outputs[m].shape)
    # print(to_return)
    # print(to_return[0].shape)
    # to_return.clear()
    # print(len(act[0]))
    # for i in range(len(act[0])):
    #     print(act[0][i].shape)
    model_inputs.clear()





result = [torch.stack(tensors).sum(dim=0) for tensors in zip(*layers_act)]
result = [(tensor / 2) for tensor in result]
for index_1, elem in enumerate(result):
    top_values, top_indices = torch.topk(elem, k=10)
    tenth_ave_act.append([(index_1, top_indices[i].item()) for i, index in enumerate(top_indices)])
print(tenth_ave_act)
            

