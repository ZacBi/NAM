import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tempfile
from share_btn import community_icon_html, loading_icon_html, share_js, save_js
import huggingface_hub
import gradio as gr
import os
# from gill import utils
# from gill import models
# 没有用到呀
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
# HF Transfer 功能允许用户在 Hugging Face Hub 上缓存模型，并在需要时自动从缓存中加载模型。
# 禁用这个功能意味着在加载模型时，不会使用 HF Transfer 功能，而是直接从远程下载模型
from modelscope.hub.snapshot_download import snapshot_download

from detectron2.config import get_cfg, LazyConfig
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "False"
def upload_image(state, image_input):
    conversation = state[0]
    chat_history = state[1]
    # 使用 PIL 库中的 Image.open() 方法打开上传的图像文件，
    # 然后调用 resize() 方法将图像大小调整为 (224, 224) 像素，再调用 convert('RGB') 方法将图像转换为 RGB 模式
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    # 保存处理后的图像，覆盖原图像文件。这一步可能是为了减少图像尺寸，以节省存储空间或加快处理速度
    input_image.save(image_input.name)  # Overwrite with smaller image.
    # 处理后的图像路径以 <img> 标签的形式添加到对话记录中，该标签用于将图像显示在对话界面中
    # 以html格式处理的
    conversation += [(f'<img src="./file={image_input.name}" style="display: inline-block;">', "")]
    # 返回更新后的状态和对话记录。l
    return [conversation, chat_history + [input_image, ""]], conversation


def reset():
    return [[], []], []


def reset_last(state):
    # 使用切片操作 [:-1]，将最后一条对话从对话记录中删除，得到更新后的对话记录 conversation
    conversation = state[0][:-1]
    # 使用切片操作 [:-2]，将最后两条聊天历史记录从聊天历史中删除，得到更新后的聊天历史记录 chat_history
    chat_history = state[1][:-2]
    return [conversation, chat_history], conversation

# 这个是从网上保存生成的图像
def save_image_to_local(image: Image.Image):
    # TODO(jykoh): Update so the url path is used, to prevent repeat saving.
    # 更新代码，使用 URL 路径来保存图像，避免重复保存
    # tempfile._get_candidate_names() 是 Python 中 tempfile 模块内部的一个函数，主要用于生成临时文件名的候选列表
    # 使用 next() 函数获取其中的第一个文件名
    filename = next(tempfile._get_candidate_names()) + '.png'
    # 将图像 image 保存到这个生成的文件名中
    image.save(filename)
    return filename

# 输入：
# 输入文本
# state: 表示模型的当前状态，可能包含了模型内部的一些信息或者状态
# ret_scale_factor: 表示返回的缩放因子，用于调整生成文本的返回长度
# num_words: 表示生成文本的长度或者单词数目
# temperature: 表示控制生成文本的多样性和创造性的温度参数。较高的温度会导致更加随机和多样化的生成结果，而较低的温度则会倾向于生成更加确定性和传统的文本
def find_token(model, input, input_text, list, model_outputs):
    code = model.model.tokenizer.encode(input, add_special_tokens=False)[0]
    # 编码完是包含初始的特殊标记：token_id=2, 10是表示10个输入的样本
    print("code:", code)
    layers_target = []
    code_text = model.model.tokenizer.encode(model_outputs[0])
    print("code_text: ", code_text)
    code_model_text = model.model.tokenizer.encode(input_text)
    print(len(code_model_text))
    code_id = [i for i, token_id in enumerate(code_text) if token_id == 2335]
    if len(code_id)==0:
        print("there is no target!!")
        pass
    else: 
        # 比如按照现在的输入，初始输入是torch。size(1,14,D)，加上最开始的特殊标记torch.size(1,15,D)
        # 如果直接生成了dog，那么dog的token就是在torch.size(1,16,D), 也就是索引15，target_id表示要找的单词（例如“dog”的索引）
        target_id = len(code_model_text) + 4 + code_id[0]
        for i in range(32):
            layers_target.append(list[i][target_id][-1,:]) # act[i]
        return layers_target
# 获得一系列的矩阵Wout*Wreadout
def get_WoutWread(model):
    project = []
    Wreadout_t = model.model.lm.lm_head.weight.t()
    for i in range(32):
        Wout_t = model.model.lm.model.decoder.layers[i].fc2.weight.t()
        proj = Wout_t.mm(Wreadout_t)
        project.append(proj)
    return project
# 金毛法(代简化todo：增加张量维度，减少列表使用)
# 计算得分前十
def find_proj(model, layers_list, token_id):
    project = []
    scores = []
    tenth_ave_act =[]
    project = get_WoutWread(model) # 32个矩阵 layers_list也是有32个元素的列表
    for index, layerlist in enumerate(layers_list): # 例如layers_list[0]存储的是所有输入通过第一个FFN的激活函数，生成‘dog’token时的激活值;layerlist是一个列表
        score = [layerlist[i]*project[index][:,token_id] for i in len(layerlist)]
        scores.append(score) # scores[0]里面存储的就是第0个FFN计算得分
    result = [torch.stack(tensors).sum(dim=0) for tensors in zip(*scores)]
    result = [(tensor / len(score)) for tensor in result]
    for index_1, elem in enumerate(result):
        top_values, top_indices = torch.topk(elem, k=10)
        tenth_ave_act.append([(index_1, top_indices[i].item()) for i, index in enumerate(top_indices)])
    print(tenth_ave_act)
def deal_img(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)  # 拼接文件路径
            input_image = Image.open(file_path).resize(
                (224, 224)).convert('RGB')
            input_image.save(file_path)
            files_list.append(input_image)  # 将文件路径添加到列表中
    return files_list
def find_eva_act(model, path, input_text, ret_scale_factor, num_words, temperature):
    print(type('Q: '))
    print(type('\nA:'))
    input_text = str(input_text)
    print(type(input_text))
    input_prompt = 'Q: ' + input_text + '\nA:'
    to_return = []
    to_return_1 = []
    model_inputs = []
    model_outputs = []
    tenth_ave_act = []
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    files_list = deal_img(path)
    layers_act = [[] for _ in range(len(files_list))]
    tenth_max_act = [[] for _ in range(len(files_list))]
    model_input_s = [[] for _ in range(len(files_list))]
    for i, lst in enumerate(model_input_s):
        lst.append(files_list[i])
        lst.append(input_prompt)
    model_input_s = [s for s in model_input_s if s != '']
    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95
    act = [[] for _ in range(32)]
    def forward_hook(n):
        def fn(_, input, output):
            act[n].append(output.detach())
        return fn
    handle_act = [model.model.lm.model.decoder.layers[n].activation_fn.register_forward_hook(forward_hook(n)) for n in
              range(32)]
    for index, model_input in enumerate(model_input_s): 
        with torch.no_grad():
            print('Running model.generate_for_images_and_texts with', model_inputs, flush=True)
            model_outputs = model.generate_for_images_and_texts(model_input_s[index],
                                                                num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                                temperature=temperature, max_num_rets=1,
                                                                num_inference_steps=50, generator=g_cuda)
            print(model_outputs)
            # act[i]里面储存的是第i层的激活值输出：[torch.size(1,14,4096),torch.size(1,15,4096),……]这种；
            # layers_act[i]里面存储的是：layers_target.append(act[j][target_id][-1,:]),index的model_input,即将输出'dog'token时，每一层的激活值的最后一行激活值；
            layers_act[index] = find_token(model, "dog", input_text, act, model_outputs)
    # 对处理结果求平均
    act_sum = [torch.stack(tensors).sum(dim=0) for tensors in zip(*layers_act)]
    act_ave = [(tensor / len(files_list)) for tensor in act_sum]
    for index_1, elem in enumerate(act_ave):
        top_values, top_indices = torch.topk(elem, k=10)
        tenth_ave_act.append([(index_1, top_indices[i].item()) for i, index in enumerate(top_indices)])
    print(tenth_ave_act)
def save_output_img(model_outputs, save_path, index):
    response = ''
    text_outputs = []
    flag = 0
    for output_i, p in enumerate(model_outputs):
        if type(p) == str:
            # Remove the image tokens for output.
            text_outputs.append(p.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', ''))
            response += p
        elif type(p) == dict:
            flag = 1
            # Decide whether to generate or retrieve.
            if p['decision'] is not None and p['decision'][0] == 'gen':
                image = p['gen'][0][0]#.resize((224, 224))
                print(image)
                image.show()
                image.save(os.path.join(save_path, "dog") + "{}.jpg".format(index)) 
            else:
                image = p['ret'][0][0]#.resize((224, 224))
                print(image)
                image.show()
                image.save(os.path.join(save_path, "dog") + "{}.jpg".format(index)) 
    if flag == 1:
        return [response, image]
    else:
        return [response]

def generate_for_prompt(model, input_text, model_inputs, ret_scale_factor, num_words, temperature):
    # torch.Generator 类来创建一个生成器对象 g_cuda，并指定该生成器在 CUDA 设备上进行操作，并手动设置种子为 1337
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    # Ignore empty inputs.
    # 当输入文本为空时，直接返回当前的对话状态和内容，并且可能更新界面元素为可见状态，避免进行后续的处理，从而忽略空输入并快速返回结果
    #if len(input_text) == 0:
    #    return state, state[0], gr.update(visible=True)
    # todo：这里应该需要规定input_text="What is this?"
    input_prompt = 'Q: ' + input_text + '\nA:'
    # conversation = state[0]
    # chat_history = state[1]
    # print('Generating for', chat_history, flush=True)
    
    # If an image was uploaded, prepend it to the model. 如果上传了一张图片，将其添加到模型前面
    # 这里我的理解是chat_history其实就是上传的图片，历史的聊天记录是图片
    # model_inputs = chat_history
    model_inputs.append(input_prompt)
    # Remove empty text.
    model_inputs = [s for s in model_inputs if s != '']
    # 存在一定的温度调节生成文本的情况下，top_p 的值被调整为 0.95。这可以影响模型生成文本时对词汇的采样方式，进而影响生成文本的多样性和创造性。
    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95
    # code_2 = model.model.tokenizer.encode(model_inputs[1])

    print('Running model.generate_for_images_and_texts with', model_inputs, flush=True)
    model_outputs = model.generate_for_images_and_texts(model_inputs,
                                                        num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                        temperature=temperature, max_num_rets=1,
                                                        num_inference_steps=50, generator=g_cuda)
    print('model_outputs', model_outputs, ret_scale_factor, flush=True)
    tokenized_text = model.model.tokenizer.tokenize(model_outputs[0])
    code = model.model.tokenizer.encode(model_outputs[0]) 
    print(tokenized_text)
    print(code)
    return model_outputs 



