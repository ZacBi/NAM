# 图像归因
# 我现在可以做的是纯激活值（分为IMG0和IMG0-IMG7)
# 先做IMG0-IMG7的纯激活值
# 32：16次生IMG0-IMG7 17,18
from transformers import AutoTokenizer
import torch
import glob
import os
import matplotlib.pyplot as plt
from typing import List, Union


base_path = '/data/ruip/eva02/gill_done/picture_data/output'
pattern = 'dog_*/layers_act_tensor_dog*.pt'
save_dir = '/data/ruip/eva02/gill_done/picture_attri_output'
path_wout = '/data/ruip/eva02/gill_done/Wout_Wreadout/Wout.pt'
path_wreadout = '/data/ruip/eva02/gill_done/Wout_Wreadout/Wreadout.pt'
file_paths = glob.glob(os.path.join(base_path, pattern))


# 这里更应该些写成layers的形式
def get_WoutWread(model):
    Wreadout_t = model.lm.lm_head.weight.t()
    # Assuming you are using the first layer for simplicity
    Wout_t = model.lm.model.decoder.layers[0].fc2.weight.t()
    # Add a new dimension at the beginning
    proj = Wout_t.mm(Wreadout_t).unsqueeze(0)
    for i in range(1, 32):  # todo 优化
        Wout_t = model.lm.model.decoder.layers[i].fc2.weight.t()
        proj_i = Wout_t.mm(Wreadout_t).unsqueeze(0)
        # Concatenate along the new dimension
        proj = torch.cat((proj, proj_i), dim=0)
    return proj


def plot_overlap_counts(top_k_indices, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for k, (unique_indices_2d, counts) in top_k_indices.items():
        # 统计从1到10的数量
        count_values = torch.bincount(counts, minlength=11)[1:11]

        plt.figure(figsize=(10, 5))
        plt.bar(range(1, 11), count_values.cpu(
        ).float().numpy(), label=f'Top {k}')
        plt.xlabel('Count Value')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Count Values for Top {k}')
        plt.legend()
        save_path = os.path.join(save_dir, f'MAX.png')
        plt.savefig(save_path)
        plt.close()


# 只处理一个权重的
def pic_value_mean(path, save_dir):
    # 加载并处理张量
    tensors = [torch.unsqueeze(torch.load(file_path), 0)
               for file_path in file_paths]
    stack_tensor = torch.cat(tensors, dim=0).permute(1, 0, 2, 3)
    stack_tensor = torch.mean(stack_tensor, dim=0)  # (10,32,16384)
    # 获取前50, 前100, 前500个元素的索引
    top_k_indices = {}
    for k in [50, 100, 500]:
        top_k_indices[k] = []

        for tensor in stack_tensor:
            values, indices = torch.topk(tensor.view(-1), k)
            top_k_indices[k].append(indices)

        # 合并所有索引并统计重合次数
        # print(top_k_indices[k][0])
        all_indices = torch.cat(top_k_indices[k])
        unique_indices, counts = all_indices.unique(return_counts=True)
        row_indices = unique_indices // stack_tensor.shape[2]
        col_indices = unique_indices % stack_tensor.shape[2]
        unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
        top_k_indices[k] = (unique_indices_2d, counts)
    # 打印前50个最大值的索引及其重合次数
    plot_overlap_counts(top_k_indices, save_dir)

# 只统计IMG0=


def pic_value_img0(paths, save_dir):
    # 加载并处理张量
    tensors = [torch.unsqueeze(torch.load(file_path), 0)
               for file_path in file_paths]  # （8，32，16384）
    stack_tensor = torch.cat(tensors, dim=0).permute(
        1, 0, 2, 3)  # (8,10,32,16384)
    stack_tensor = stack_tensor[0]  # (10,32,16384)
    # 获取前50, 前100, 前500个元素的索引
    top_k_indices = {}
    for k in [50, 100, 500]:
        top_k_indices[k] = []
        # 单批次·
        for tensor in stack_tensor:
            values, indices = torch.topk(
                tensor.view(-1), k)  # （32，16384）->(32*16384)
            top_k_indices[k].append(indices)

        # 合并所有索引并统计重合次数
        # print(top_k_indices[k][0])
        all_indices = torch.cat(top_k_indices[k])
        unique_indices, counts = all_indices.unique(return_counts=True)
        # todo:修改
        row_indices = unique_indices // stack_tensor.shape[2]
        col_indices = unique_indices % stack_tensor.shape[2]
        unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
        top_k_indices[k] = (unique_indices_2d, counts)
    # 打印前50个最大值的索引及其重合次数
    plot_overlap_counts(top_k_indices, save_dir)
# 激活值×Wout


def find_proj(model, layers_list, token_id):
    project = get_WoutWread(model)
    project = project.permute(2, 0, 1)
    project_new = project[token_id]
    expanded_project = torch.cat(
        [project_new.unsqueeze(0)] * layers_list.shape[0], dim=0)
    # scores = torch.zeros((layers_list.shape[0], layers_list.shape[1], layers_list.shape[2])).half()
    expanded_project = torch.nn.Parameter(expanded_project)
    if torch.cuda.is_available():
        expanded_project = expanded_project.cuda()
    else:
        print("no device is avail")
    print(layers_list.device)
    print(expanded_project.device)
    scores = layers_list*expanded_project
    mean_scores = torch.mean(scores, dim=0)  # (32, 16384)
    top_values, top_indices = torch.topk(mean_scores, k=10, dim=1)
    print(top_indices)
# def pic_value_Wout(path,save_dir,path_wout):
#     tensors = [torch.unsqueeze(torch.load(file_path), 0) for file_path in file_paths]
#     stack_tensor = torch.cat(tensors, dim=0).permute(1, 0, 2, 3)
#     stack_tensor = torch.mean(stack_tensor, dim=0) # (10,32,16384)
#     wout = torch.load(path_wout) # (32,16384,4096)
#     for i in range(10):
#         expanded_project_i = torch.unsqueeze(torch.sum(stack_tensor[i]*wout,dim=-1),0) #(1,32,16384,4096)
#         if i==1:
#             expanded_project = expanded_project_i
#         else:
#             expanded_project = torch.cat(expanded_project,expanded_project_i)
#     # 得到的是(10,32,16384)
#     top_k_indices = {}
#     for k in [50, 100, 500]:
#         top_k_indices[k] = []

#         for tensor in expanded_project:
#             values, indices = torch.topk(tensor.view(-1), k)
#             top_k_indices[k].append(indices)

#         # 合并所有索引并统计重合次数
#         # print(top_k_indices[k][0])
#         all_indices = torch.cat(top_k_indices[k])
#         unique_indices, counts = all_indices.unique(return_counts=True)
#         row_indices = unique_indices // stack_tensor.shape[2]
#         col_indices = unique_indices % stack_tensor.shape[2]
#         unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
#         top_k_indices[k] = (unique_indices_2d, counts)
#     # 打印前50个最大值的索引及其重合次数
#     plot_overlap_counts(top_k_indices,save_dir)


def pic_value_Wout(path, save_dir, path_wout):
    tensors = [torch.unsqueeze(torch.load(file_path), 0)
               for file_path in file_paths]
    stack_tensor = torch.cat(tensors, dim=0).permute(1, 0, 2, 3)
    stack_tensor = torch.mean(stack_tensor, dim=0)  # (10,32,16384)

    wout = torch.load(path_wout)  # (32,16384,4096)
    if torch.cuda.is_available():
        wout = wout.cuda()
    else:
        print("no device is avail")
    expanded_project = []
    print(stack_tensor.device)
    print(wout.device)
    for i in range(10):
        expanded_project_i = torch.sum(
            stack_tensor[i].unsqueeze(-1) * wout, dim=-1).unsqueeze(0)  # (1,32,16384)
        expanded_project.append(expanded_project_i)

    expanded_project = torch.cat(expanded_project, dim=0)  # (10,32,16384)
    top_k_indices = {}
    for k in [50, 100, 500]:
        top_k_indices[k] = []

        for tensor in expanded_project:
            values, indices = torch.topk(tensor.view(-1), k)
            top_k_indices[k].append(indices)

        # 合并所有索引并统计重合次数
        all_indices = torch.cat(top_k_indices[k])
        unique_indices, counts = all_indices.unique(return_counts=True)
        row_indices = unique_indices // stack_tensor.shape[2]
        col_indices = unique_indices % stack_tensor.shape[2]
        unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
        top_k_indices[k] = (unique_indices_2d, counts)
    # 打印前50个最大值的索引及其重合次数
    plot_overlap_counts(top_k_indices, save_dir)


def pic_value_Wout0(path, save_dir, path_wout, topk):
    tensors = [torch.unsqueeze(torch.load(file_path), 0)
               for file_path in file_paths]
    # filp batch_size and img_token_num
    stack_tensor = torch.cat(tensors, dim=0).permute(
        1, 0, 2, 3)  # (8,10,32,16384)
    stack_tensor = stack_tensor[0]  # (10,32,16384)

    wout = torch.load(path_wout)  # (32,16384,4096)
    if torch.cuda.is_available():
        wout = wout.cuda()
    else:
        print("no device is avail")
    expanded_project = []
    print(stack_tensor.device)
    print(wout.device)
    for i in range(10):
        expanded_project_i = torch.sum(
            stack_tensor[i].unsqueeze(-1) * wout, dim=-1).unsqueeze(0)  # (1,32,16384)
        expanded_project.append(expanded_project_i)

    expanded_project = torch.cat(expanded_project, dim=0)  # (10,32,16384)
    top_k_indices = {}
    for k in [50, 100, 500]:
        top_k_indices[k] = []

        for tensor in expanded_project:
            values, indices = torch.topk(tensor.view(-1), k)
            top_k_indices[k].append(indices)

        # 合并所有索引并统计重合次数
        all_indices = torch.cat(top_k_indices[k])
        unique_indices, counts = all_indices.unique(return_counts=True)
        row_indices = unique_indices // stack_tensor.shape[2]
        col_indices = unique_indices % stack_tensor.shape[2]
        unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
        top_k_indices[k] = (unique_indices_2d, counts)
    # 打印前50个最大值的索引及其重合次数
    plot_overlap_counts(top_k_indices, save_dir)
# 利用后续归因得到的分数


def meaning(path, Wreadout_Wout_path, wout_path):  # (32,16384,16384) 表示的是分数第二维：表示16384个词的得分
    # (32,16384,16384) 第二维表示神经元，表示的是分数第三维：表示16384个词的得分
    W = torch.load(Wreadout_Wout_path)
    tensors = [torch.unsqueeze(torch.load(file_path), 0)
               for file_path in file_paths]
    stack_tensor = torch.cat(tensors, dim=0).permute(1, 0, 2, 3)
    stack_tensor = stack_tensor[0]  # (10,32,16384)
    stack_tensor = torch.mean(stack_tensor, dim=0)  # （32，16384）
    print("stack_tensor:", stack_tensor.size())
    # 第1，5，10神经元对应的词义
    top_k_indices = []  # (10,32,16384)
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/ruip/eva02/gill_done/transformer_cache/opt")  # 分词器
    values, indices = torch.topk(stack_tensor.view(-1), 5)  # （32，16384）中前5个最大值
    row_indices = indices // 16384
    col_indices = indices % 16384
    top_k_indices = torch.stack(
        (row_indices, col_indices), dim=0)  # 神经元索引（层数+该层索引）
    for i in range(top_k_indices.shape[1]):  # 5:0-4
        values, indices = torch.topk(
            W[top_k_indices[0][i]][top_k_indices[1][i]], 5)
        word_id = indices % 16384
        print("indices:", indices)
        print("valuesize:", values.size())
        word_0 = tokenizer.decode(word_id[0])
        word_1 = tokenizer.decode(word_id[1])
        word_2 = tokenizer.decode(word_id[2])
        word_3 = tokenizer.decode(word_id[3])
        word_4 = tokenizer.decode(word_id[4])
        print("top_word:", word_0, word_1, word_2, word_3, word_4)
    # 合并所有索引并统计重合次数
    # print(top_k_indices[k][0])
    # all_indices = torch.cat(top_k_indices)
    print(top_k_indices.size())  # (2,5)
    # unique_indices, counts = all_indices.unique(return_counts=True)
    # row_indices = unique_indices // stack_tensor.shape[2]
    # col_indices = unique_indices % stack_tensor.shape[2]
    # unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
    # top_k_indices = (unique_indices_2d, counts)

# meaning(file_paths,path_wreadout, path_wout)


def getWout(model):
    # 提取所有解码器层的输出权重
    weights = [layer.fc2.weight.t() for layer in model.lm.model.decoder.layers]
    # 将所有权重堆叠成一个张量
    Wout = torch.stack(weights)
    return Wout  # 形状为（32, 16384, 4096）
# act: (8, 32, 16384) 8:nums of token


def normalize_tensor(tensor):
    """Normalize a tensor to the range [0, 1]."""
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)


def calculate_scores(Wout, weights, acts):  # 如果输入的acts的结构是(10,8,32,16384)
    """Calculate and sum up the scores."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Wout = Wout.to(device)
    weights = weights.to(device)
    acts = acts.to(device)

    min_acts = torch.min(acts, dim=-1).values
    max_acts = torch.max(acts, dim=-1).values
    norm_acts = (acts - min_acts.unsqueeze(-1).expand_as(acts)) / \
        (max_acts - min_acts).unsqueeze(-1).expand_as(acts)
    weights = weights.unsqueeze(-2).unsqueeze(-3)
    Wout = Wout.unsqueeze(0).unsqueeze(0)
    norm_acts = norm_acts.unsqueeze(-1)
    # (10,8,32,16384,1)*(1,1,32,16384,4096):(10,8,32,16384,4096)*(10,8,1,1,4096)=(10,8,32,16384)
    score = torch.sum(norm_acts * Wout * weights, dim=-1)
    # scores_sum = score.sum(dim=0)
    scores_sum = score.mean(dim=1)  # (10,8,32,16384)
    # scores_sum：(8，32，16384), 这里输出的结果应该是(10,32,16384)
    return scores_sum


def pic_attr(file_path, weight, act):  # 直接从路径上加载Wout
    Wout = torch.load(file_path)
    result = calculate_scores(Wout, weight, act)
    return result
# max方法


def pic_value_mean(act):
    # 加载并处理张量
    # tensors = [torch.unsqueeze(torch.load(file_path), 0) for file_path in file_paths]
    stack_tensor = torch.cat(tensors, dim=0).permute(1, 0, 2, 3)
    stack_tensor = torch.mean(stack_tensor, dim=0)  # (10,32,16384)


def max_both(model, weight, act):
    score_1 = pic_attr(model, weight, act)
    score_2 = pic_value_mean(act)
    score = torch.max(score_1, score_2)
    # 得到最后的结果


def max_IMG(path, weight, act, topK: Union[float, List[float]] = 0.1):
    score_1 = pic_attr(path, weight, act)
    score_2 = pic_value_mean(act)
    score = torch.max(score_1, score_2)  # (10,32,16384)
    top_k_indices = {}
    for k in [50, 100, 500]:
        top_k_indices[k] = []
        for tensor in score:
            values, indices = torch.topk(tensor.view(-1), k)
            top_k_indices[k].append(indices)

        # 合并所有索引并统计重合次数
        all_indices = torch.cat(top_k_indices[k])
        unique_indices, counts = all_indices.unique(return_counts=True)
        row_indices = unique_indices // stack_tensor.shape[2]
        col_indices = unique_indices % stack_tensor.shape[2]
        unique_indices_2d = torch.stack((row_indices, col_indices), dim=1)
        top_k_indices[k] = (unique_indices_2d, counts)
    # 打印前50个最大值的索引及其重合次数
    plot_overlap_counts(top_k_indices, save_dir)


tensors = [torch.unsqueeze(torch.load(file_path), 0)
           for file_path in file_paths]
tensor = torch.cat(tensors, dim=0)
weight = torch.ones(10, 8, 4096)
max_IMG(path_wout, weight, tensor)
