import os
from typing import List

import torch
from PIL import Image

import models
from transformers import OPTForCausalLM


# Load the model correctly
path = 'path to gill_done/checkpoints'
model = models.load_gill(path)


def generate_for_prompt(input_text, model_inputs, ret_scale_factor, num_words, temperature=0.95, seed=1):
    """
    使用给定模型生成响应文本。

    :param input_text: 输入文本
    :param model: 模型实例
    :param tokenizer: 分词器实例
    :param scale_factor: 返回结果的比例因子
    :param word_count: 输出单词数量
    :param temperature: 温度参数，默认为0.95
    :param seed: 随机种子，默认为1
    :return: 生成的输出文本列表
    """

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=device).manual_seed(seed)

        prompt = f"Q: {input_text}\nA:"

        # 移除了model_inputs列表的使用，直接处理prompt
        outputs = model.generate(prompt,
                                 num_words=max(num_words, 1),
                                 ret_scale_factor=ret_scale_factor,
                                 top_p=1.0,
                                 temperature=temperature,
                                 max_num_rets=1,
                                 num_inference_steps=50,
                                 generator=generator)

        encoded_output = model.model.tokenizer.encode(outputs[0])
        print(f"Generated output ID: {encoded_output}")

        return outputs

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return []

# get matrix
def get_WoutWread(gill_model):
    Wreadout_t = gill_model.lm.lm_head.weight.t()
    # Collect all weights from each layer into one tensor
    Wouts_t = torch.stack([
        layer.fc2.weight.t()
        for layer in gill_model.lm.model.decoder.layers
    ])

    # Perform batch matrix multiplication
    proj = torch.bmm(Wouts_t, Wreadout_t.unsqueeze(0))

    # Remove the unnecessary dimension added by unsqueeze
    return proj.squeeze(1)


def get_all_layer_activations_for_target_token(target_token_id, input_text, layer_activations, model_output):
    """
    根据输入的token在模型输出中找到对应位置并获取各层激活值

    参数：
    - input_token: 需查找的token对应的id
    - input_text: 输入文本（用于调试信息）
    - layer_activations: 模型各层的激活值列表
    - model_output: 模型的输出结果

    返回：
    - layers_target: 包含目标token在所有层的激活值的张量
    """

    encode_token_ids = model.model.tokenizer.encode(model_output[0])

    # 定位目标token的位置
    target_positions = [
        index - 1 for index, token_id in enumerate(encode_token_ids)
        if token_id == target_token_id
    ]

    if not target_positions:
        print(f"在{input_text}中未能找到目标token!")
        return None

    # 收集所有层的激活值
    target_idx = target_positions[0]
    all_activations = [
        activation[target_idx][-1, :].unsqueeze(0)
        for activation in layer_activations[:]
    ]

    # 合并所有层的激活值
    layers_target = torch.cat(all_activations, dim=0)

    print(f"合并后的激活值维度：{layers_target.size()}")
    return layers_target


# 示例调用
# find_token(token_id, "示例文本", activations_list, model_outputs)


def resize_and_convert_image(file_path: str, width: int = 224, height: int = 224, format='RGB') -> Image.Image:
    """
    Resize and convert an image to RGB format.

    Args:
        file_path (str): The path to the image file.

    Returns:
        Image.Image: The processed image object.
    """
    try:
        img = Image.open(file_path)
        resized_img = img.resize((width, height))
        rgb_img = resized_img.convert(format)
        return rgb_img
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def deal_img(image_dir: str) -> List[Image.Image]:
    """
    Handle images for matching the input of Gill Model.

    Args:
        image_dir (str): Directory containing the images.

    Returns:
        List[Image.Image]: A list of processed image objects.
    """
    files_list = [os.path.join(root, file)
                  for root, dirs, files in os.walk(image_dir)
                  for file in files]

    images = [resize_and_convert_image(
        path) for path in files_list if path.endswith(('.jpg', '.jpeg', '.png'))]

    # Filter out any None values that may have been returned due to errors
    return [img for img in images if img is not None]

# Take the larger value of the normalization results of the two methods to determine the degree of neuron activation
# Model is Gill's opt model
# layers_act is the activation value for each layer
# token_id: The ID of the subject in the word list


def max_of_both(model, layers_act, target_cls_token_id):
    """
    model: Gill's opt model, backbone
    layers_act: Activation value: (num_layers, 32, 16384)
    target_cls_token_id: The ID of the subject in the word list
    """
    project = get_WoutWread(model).permute(2, 0, 1)
    project_new = project[target_cls_token_id]
    num_layers = layers_act.shape[0]

    # Efficiently expand project_new to match layers_act shape
    expanded_project = project_new.expand(num_layers, -1, -1)

    # Move tensors to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    expanded_project = expanded_project.to(device)
    layers_act = layers_act.to(device)

    scores = layers_act * expanded_project

    # Compute normalization parameters once per layer
    max_values = torch.max(scores, dim=-1)[0]
    min_values = torch.min(scores, dim=-1)[0]
    max_values_act = torch.max(layers_act, dim=-1)[0]
    min_values_act = torch.min(layers_act, dim=-1)[0]

    # Normalize scores and activations
    def normalize(tensor, min_val, max_val):
        """Normalize a tensor between min and max values."""
        return (tensor - min_val) / (max_val - min_val)

    normalized_scores = normalize(
        scores, min_values.unsqueeze(-1), max_values.unsqueeze(-1))
    normalized_list = normalize(
        layers_act, min_values_act.unsqueeze(-1), max_values_act.unsqueeze(-1))

    # Find maximum between normalized scores and activations
    max_neuron_val = torch.max(normalized_list, normalized_scores)

    # Average over neurons to obtain final result
    max_neuron = torch.mean(max_neuron_val, dim=(1, 2))

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
    获取所有图像对于给定提示的所有层激活值。

    参数：
    - image_dir: 图像目录路径
    - prompt: 要生成的提示
    - ret_scale_factor: 模型输出的比例因子
    - num_words: 要生成的单词数
    - temperature: softmax 的温度
    - num_llm_layers: LLM 中的层数
    - target_cls_token_id: 目标分类令牌ID（例如，“狗”的索引为2335）

    返回：
    - 所有图像的激活张量
    """

    try:
        image_list = deal_img(image_dir=image_dir)

        # 初始化一个空列表用于存储每张图片的激活值
        activations = []

        # 定义前向钩子函数
        def forward_hook(n):
            def fn(_, __, output):
                act[n].append(output.detach())
            return fn

        # 遍历图像列表
        for img in image_list:
            model_inputs = [img]

            # 初始化激活值列表
            act = [[] for _ in range(num_llm_layers)]

            # 注册前向钩子
            hooks = [
                model.model.lm.model.decoder.layers[layer_idx].activation_fn.register_forward_hook(
                    forward_hook(layer_idx))
                for layer_idx in range(num_llm_layers)
            ]

            # 使用模型生成结果
            with torch.no_grad():
                model_outputs = generate_for_prompt(prompt, model_inputs,
                                                    ret_scale_factor=ret_scale_factor,
                                                    num_words=num_words,
                                                    temperature=temperature)

                # 获取目标令牌的所有层激活值
                layers_act = get_all_layer_activations_for_target_token(
                    target_cls_token_id, prompt, act, model_outputs)

                # 移除注册的钩子
                for h in hooks:
                    h.remove()

                # 将当前图像的激活值添加到列表中
                if layers_act is not None:
                    activations.append(layers_act.unsqueeze(0))

        # 如果有有效的激活值，则堆叠它们并返回；否则抛出异常
        if activations:
            return torch.cat(activations, dim=0)
        else:
            raise ValueError("No valid activations found.")

    except Exception as e:
        print(f"An error occurred while processing images: {e}")
        return None


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
    llm_backbone = OPTForCausalLM.from_pretrained(
        pretrained_model_name_or_path=backone_model_path)
    scores_for_all_neurons_to_target_cls = max_of_both(
        llm_backbone, layers_act, target_cls_token_id)

    # load model and save
    save_path = 'your path to save'
    torch.save(scores_for_all_neurons_to_target_cls, save_path)
