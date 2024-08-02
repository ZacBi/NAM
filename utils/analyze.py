
from dataclasses import dataclass
import json
import torch
from typing import List, Tuple
import os

# 定义数据行类，用于存储每条记录的信息


@dataclass
class DataRow:
    """数据行类，包含ID、提示信息、图片路径、归因路径及形状信息"""
    id: int
    prompt: str
    pic_pth: str
    attribution_pth: str
    shape: Tuple[int, ...]

# 定义统计行类，用于存储每层的统计信息


@dataclass
class StatisticRow:
    """统计行类，包含层索引、比例、前k值、唯一值和计数等信息"""
    layer_idx: int
    ratio: float
    top_k: int
    neuron_indices: torch.Tensor
    counts: torch.Tensor
    norm_counts: torch.Tensor


def load_json_as_class_list(json_file_path: str, cls) -> List:
    """
    将JSON文件加载并转换为指定类的对象列表
    Args:
        json_file_path (str): JSON文件路径
        cls (type): 要转换成的对象类型
    Returns:
        list: 对象列表
    """
    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        return [cls(**item) for item in json_data]
    except Exception as e:
        print(f"Error loading {json_file_path}: {e}")
        return []


# 需要考虑如何计算n个向量(集合)的相似度
def process_attribution(data_rows: List[DataRow], ratio_list: torch.Tensor = torch.arange(0.1, 1.1, 0.1)) -> None:
    """
    处理归因数据并生成统计信息
    Args:
        data_rows (List[DataRow]): 数据行列表
    """
    if not data_rows:
        raise ValueError('未找到任何数据权重行')

    # num_tokens = number of img[0 - 8] = 8
    num_tokens, num_layers, dim_ffn = data_rows[0].shape

    # 加载所有归因数据并仅保留img0的归因, take attribution of img0

    attribution = [torch.load(row.attribution_pth)[0] for row in data_rows]
    # default : dim = 0, (batch_size, num_layer, dim_h)
    attribution = torch.stack(attribution)
    # sorted attributiion, indices shape: (batch_size, num_layer, dim_h), value of each cell is index(int)
    _, indices = torch.sort(attribution, dim=-1)

    seq_statistics = []
    ratio_layer_stat_table = {}
    for ratio in ratio_list:
        top_k = int(dim_ffn * ratio)
        for layer_idx in range(num_layers):
            top_k_attribution = attribution[:, layer_idx, :top_k].view(-1)

            # 直接获取唯一值及其出现次数，无需额外排序步骤 [[1, 2], [2, 3]] = [1, 2, 3], [1, 2, 1]
            # 取batch中,每一层神经元中,索引的计数
            unique_values, counts = torch.unique(
                top_k_attribution, return_counts=True)

            # 将值和计数配对，并按计数降序排序
            # 注意：这里先将unique_values和counts转换为tuple列表，然后排序，最后再分开
            value_count_pairs = list(
                zip(unique_values.tolist(), counts.tolist()))
            sorted_pairs = sorted(
                value_count_pairs, key=lambda x: x[1], reverse=True)

            # 分离排序后的值和计数
            sorted_values, sorted_counts = zip(*sorted_pairs)

            # 创建统计行实例并添加到列表
            stat_row = StatisticRow(layer_idx=layer_idx,
                                    ratio=ratio,
                                    top_k=top_k,
                                    neuron_indices=sorted_values,
                                    counts=sorted_counts,
                                    norm_counts=sorted_counts / num_tokens)
            seq_statistics.append(stat_row)
            key = f'{ratio}_{layer_idx}'
            ratio_layer_stat_table[key] = stat_row

    # 保存统计信息至文件
    torch.save(seq_statistics, 'seq_statistic.pth')
    torch.save(ratio_layer_stat_table, 'statistic_table.pth')


if __name__ == '__main__':
    dir_path = 'path_to_json_config_file'
    data_rows = load_json_as_class_list(dir_path, DataRow)
    process_attribution(data_rows)
