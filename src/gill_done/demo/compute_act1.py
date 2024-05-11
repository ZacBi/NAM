import os
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "False"
# #save-btn、#share-btn 分别设置了 id 为 save-btn 和 share-btn 的按钮元素的背景渐变颜色。
css = """
    # 设置 id 为 chatbot 的元素的最小高度为 300px，确保聊天机器人界面的高度不会小于 300px。
    #chatbot { min-height: 300px; }
    # #save-btn、#share-btn 分别设置了 id 为 save-btn 和 share-btn 的按钮元素的背景渐变颜色。
    # 当鼠标悬停在按钮上时，背景颜色会发生变化，采用不同的渐变颜色。
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
    # 当鼠标悬停在按钮图片上时，取消图片的放大效果并恢复正常大小。
    #gallery { z-index: 999999; }
    #gallery img:hover {transform: scale(2.3); z-index: 999999; position: relative; padding-right: 30%; padding-bottom: 30%;}
    #gallery button img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; padding-bottom: 0;}
    # 在不支持鼠标悬停的设备上，取消图片的放大效果（transform: none），恢复到原始大小。
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
# 解释，如果用金毛的方法的话：实际上是投影矩阵torch.size([16384,50274]), 单词矩阵torch.size([50274, 4096])
# Download model from HF Hub.
# ckpt_path = huggingface_hub.hf_hub_download(
#    repo_id='jykoh/gill', filename='pretrained_ckpt.pth.tar', local_dir='/data1/ruip/gill/gill-main_1/download')
# decision_model_path = huggingface_hub.hf_hub_download(
#     repo_id='jykoh/gill', filename='decision_model.pth.tar', local_dir='/data1/ruip/gill/gill-main_1/download')
# args_path = huggingface_hub.hf_hub_download(
#    repo_id='jykoh/gill', filename='model_args.json', local_dir='/data1/ruip/gill/gill-main_1/download')
path = '/data/ruip/eva02/gill-main_2/checkpoints/gill_opt'
# model = models.load_gill('./', args_path, ckpt_path, decision_model_path)
model = models.load_gill(path)
print(model.model.lm.model.decoder.layers[0].fc1.weight.size())
print(model.model.lm.model.decoder.layers[0].fc2.weight.size())
print(model.model.lm.model.decoder.layers[3].fc1.weight.size())
print(model.model.lm.model.decoder.layers[3].fc2.weight.size())
print(model.model.lm.lm_head.weight.size())
print("models load complished\n")
# 接受两个参数 state 和 image_input
# state 是一个包含两个元素的列表，分别表示对话和聊天历史记录；image_input 是一个上传的图像文件对象
# 这段代码的作用是将上传的图像文件进行处理，更新对话记录，然后返回更新后的状态。这样就能在程序中处理上传的图像，并在对话界面中显示处理后的图像
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
def generate_for_prompt(input_text,  model_inputs, ret_scale_factor, num_words, temperature):
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
    return model_outputs
    print('model_outputs', model_outputs, ret_scale_factor, flush=True)
    tokenized_text = model.model.tokenizer.tokenize(model_outputs[0])
    code = model.model.tokenizer.encode(model_outputs[0]) 
    print(tokenized_text)
    print(code)
def get_WoutWread(model):
    project = []
    Wreadout_t = model.model.lm.lm_head.weight.t()
    for i in range(32):
        Wout_t = model.model.lm.model.decoder.layers[i].fc2.weight.t()
        proj = Wout_t.mm(Wreadout_t)
        project.append(proj)
    return project
# 金毛法(代简化todo：增加张量维度，减少列表使用)
def find_proj(model, layers_list, token_id):
    project = []
    scores = []
    project = get_WoutWread(model) # 32个矩阵
    for index, layerlist in enumerate(layers_list): # 例如layers_list[0]存储的是所有输入通过第一个FFN的激活函数，生成‘dog’token时的激活值;layerlist是一个列表
        score = [layerlist[i]*project[i][:,token_id] for i in len(project)]
        scores.append(score) # scores[0]里面存储的就是第0个FFN计算得分
    result = [torch.stack(tensors).sum(dim=0) for tensors in zip(*scores)]
    result = [(tensor / len(score)) for tensor in result]
    for index_1, elem in enumerate(result):
        top_values, top_indices = torch.topk(elem, k=10)
        tenth_ave_act.append([(index_1, top_indices[i].item()) for i, index in enumerate(top_indices)])
    print(tenth_ave_act)
def find_token(input, input_text, list, model_outputs):
    code = model.model.tokenizer.encode(input, add_special_tokens=False)[0]
    # 编码完是包含初始的特殊标记：token_id=2, 10是表示10个输入的样本
    print("code:", code)
    layers_target = []
    code_text = model.model.tokenizer.encode(model_outputs[0])
    print("code_text: ", code_text)
    code_model_text = model.model.tokenizer.encode(model_inputs[1])
    print(len(code_model_text))
    code_id = [i for i, token_id in enumerate(code_text) if token_id == 2335]
    if len(code_id)==0:
        print("there is no target!!")
        pass
    else: 
        # 比如按照现在的输入，初始输入是torch。size(1,14,D)，加上最开始的特殊标记torch.size(1,15,D)
        # 如果直接生成了dog，那么dog的token就是在torch.size(1,16,D), 也就是索引15，target_id表示要找的单词（例如“dog”的索引）
        target_id = len(code_model_text) + 2 + code_id[0]
        for i in range(32):
            layers_target.append(list[i][target_id][-1,:]) # act[i]
        return layers_target
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
# def cal_prob(input):
act = [[] for _ in range(32)]
def forward_hook(n):
    def fn(_, input, output):
        act[n].append(output.detach())
    return fn
handle_act = [model.model.lm.model.decoder.layers[n].activation_fn.register_forward_hook(forward_hook(n)) for n in
              range(32)]    
    # 返回更新后的状态和对话记录。l
path_1 = '/data/ruip/eva02/gill-main_2/test_dog_data'
input_text = "What is this?"
# 只生成文本，不生成图片的前提下
to_return = []
to_return_1 = []
model_inputs = []
model_outputs = []
# range(x)表示样本数
layers_act = [[] for _ in range(1)]
tenth_max_act = [[] for _ in range(1)]
tenth_ave_act = []
# 得到处理图像的列表
files_list = deal_img(path_1)
for i in range(1):
    print("生成第{}个".format(i))
    model_inputs.append(files_list[i])
    with torch.no_grad():
        model_outputs = generate_for_prompt(input_text, model_inputs, ret_scale_factor=1.0, num_words=32, temperature=0.0)
        if i < 2:
            # 存储得到的32个激活值编码
            layers_act[i] = find_token("dog", input_text, act, model_outputs)
    model_inputs.clear()
# 分布，对比任务，找到对应的token，dog前面的一个token;
# 如何实现找到最后一个前一个token：
# 1.生成的文本进行分词；2.找到前面一个token
# 代码功能：寻找对应的输入token：
# layers_act里面的格式是[[一维向量，……],[]]
find_proj(model, layers_act, 2335)