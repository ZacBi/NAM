import os
from typing import List

import torch
from PIL import Image

import models
from transformers import OPTForCausalLM


# Load the model correctly
path = 'path to gill_done/checkpoints'
model = models.load_gill(path)


def generate_for_prompt(input_text, model_inputs, ret_scale_factor, num_words, temperature):
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    input_prompt = 'Q: ' + input_text + '\nA:'
    model_inputs.append(input_prompt)
    # Remove empty text.
    model_inputs = [s for s in model_inputs if s != '']
    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95
    print('Running model.generate_for_images_and_texts with',
          model_inputs, flush=True)
    model_outputs = model.generate_for_images_and_texts(model_inputs,
                                                        num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                        temperature=temperature, max_num_rets=1,
                                                        num_inference_steps=50, generator=g_cuda)
    print("outputs:", model_outputs[0])
    code_text = model.model.tokenizer.encode(model_outputs[0])
    print("outputs_id:", code_text)
    return model_outputs

# get matrix
def get_WoutWread(gill_model):
    Wreadout_t = gill_model.lm.lm_head.weight.t()
    # Assuming you are using the first layer for simplicity
    Wout_t = gill_model.lm.model.decoder.layers[0].fc2.weight.t()
    # Add a new dimension at the beginning
    proj = Wout_t.mm(Wreadout_t).unsqueeze(0)
    for i in range(1, 32):
        Wout_t = gill_model.lm.model.decoder.layers[i].fc2.weight.t()
        proj_i = Wout_t.mm(Wreadout_t).unsqueeze(0)
        # Concatenate along the new dimension
        proj = torch.cat((proj, proj_i), dim=0)
    return proj  # (32,16384, 50272)


def find_token(input, input_text, list, model_outputs):
    code_text = model.model.tokenizer.encode(model_outputs[0])
    # positioning
    code_id = [i-1 for i,
               token_id in enumerate(code_text) if token_id == input]
    if len(code_id) == 0:
        print("there is no target!!")
        pass
    else:
        target_id = code_id[0]
        layers_target = list[0][target_id][-1, :]
        layers_target = layers_target.unsqueeze(0)
        for i in range(1, 32):
            layers_act_i = list[i][target_id][-1, :].unsqueeze(0)
            layers_target = torch.cat(
                (layers_target, layers_act_i), dim=0)  # act[i]
        print("layers_target:", layers_target.size())
        return layers_target  # (layers, 16384)


def deal_img(image_dir: str) -> List[Image.Image]:
    """handle image for matching the input of Gill Model"""
    files_list = []
    for root, dirs, files in os.walk(image_dir):
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
    """
    model: Gill's opt model, backbone
    layers_act: Activation value: (num_layers, 32, 16384)
    token_id: The ID of the subject in the word list
    """
    project = get_WoutWread(model)
    project = project.permute(2, 0, 1)
    project_new = project[token_id]
    expanded_project = torch.cat(
        [project_new.unsqueeze(0)] * layers_act.shape[0], dim=0)
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
        normalized_list = (
            layers_act[i] - min_value_1) / (max_value_1 - min_value_1)
        max_neuron_val = torch.max(normalized_list, normalized_scores)
        if i == 0:
            max_neuron = max_neuron_val.unsqueeze(0)
        else:
            max_neuron = torch.cat(
                (max_neuron, max_neuron_val.unsqueeze(0)), dim=0)
    # get max_neuron, Neuronal fraction obtained by the MAX method
    max_neuron = torch.mean(max_neuron, dim=0)
    return max_neuron

# model_inputs Stores images and text entered into the GILL， You'll need to organize the input yourself
# file_list = deal_img(path) Correct processing of the image gets
# input_text: Write a paragraph for each image
# layers_act: Activation value：（num, 32, 16384)

def requires_activations_for_all_tokens(image_dir: str,
                                        prompt: str,
                                        ret_scale_factor=1.3,
                                        num_words=32,
                                        temperature=0.0,
                                        num_llm_layers=32,
                                        target_cls_token_id=2335) -> torch.Tensor:
    """
    image_dir: Path to the directory containing the images
    prompt: The prompt to be generated
    ret_scale_factor: The scale factor for the output of the model
    num_words: The number of words to be generated
    temperature: The temperature for the softmax
    target_cls_token_id: the idx of 'dog' is 2335
    """
    image_list = deal_img(image_dir=image_dir)

    for i in range(len(image_list)):
        model_inputs = []
        model_inputs.append(image_list[i])
        with torch.no_grad():
            # act is the activation of all ffn layers of llm
            act = [[] for _ in range(num_llm_layers)]

            # The activation value is obtained by setting the hook function
            def forward_hook(n):
                def fn(_, input, output):
                    act[n].append(output.detach())
                return fn

            # register hook for all ffn layers
            for layer_idx in range(num_llm_layers):
                model.model.lm.model.decoder.layers[layer_idx].activation_fn.register_forward_hook(
                    forward_hook(layer_idx))

            model_outputs = generate_for_prompt(
                prompt, model_inputs, ret_scale_factor=ret_scale_factor, num_words=num_words, temperature=temperature)
            layers_act_i = find_token(
                target_cls_token_id, prompt, act, model_outputs)
            if layers_act_i is not None:
                if i == 0:
                    layers_act_i = layers_act_i.unsqueeze(0)
                    layers_act = layers_act_i
                else:
                    layers_act_i = layers_act_i.unsqueeze(0)
                    # layers_act (B, Q, T) num，
                    layers_act = torch.cat((layers_act, layers_act_i), dim=0)
        return layers_act


if __name__ == "__main__":
    # 1. image to enrich the content of style of prompt
    image_dir_path = '{your path to image dir with png of jpg format}'
    image_list = deal_img(path)

    # 2. write a paragraph to describe what you want generate
    prompt = '{Write a paragraph for each image}'

    # 3.get activation for all layers
    target_cls_token_id = 2335
    layers_act = requires_activations_for_all_tokens(
        image_dir_path, prompt=prompt, targe_cls_token_id=target_cls_token_id)

    # 4.get max_neuron, Neuronal fraction obtained by the MAX method
    backone_model_path = 'your path to llm backbone'
    llm_backbone = OPTForCausalLM.from_pretrained(pretrained_model_name_or_path=backone_model_path)
    scores_for_all_neurons_to_target_cls = max_of_both(llm_backbone, layers_act, target_cls_token_id)

    # load model and save
    save_path = 'your path to save'
    torch.save(scores_for_all_neurons_to_target_cls, save_path)
