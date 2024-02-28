#############
# MODEL ZOO #
#############

from ast import Module
from email import generator
from typing import Any
from constant import Modality
from re import L
from token import STRING
from turtle import forward
from httpx import get
import torch
from torch import nn
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer, AdapterConfig, AutoModelWithLMHead, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForPreTraining, AutoModelForCausalLM
from transformers.activations import ACT2FN

from helper import mprojector, mgenerator
modelpath = "/data/MODELS/"


class MEncoder(torch.nn.Module):
    """模态编码器"""
    def __init__(self, args):
        super().__init__()
        self.args = args

class MProjector(torch.nn.Module):
    """投影器, 用于将多模态理解器的输出投影到一个统一的空间中"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class InputProjector(MProjector):
    """输入投影器"""

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.projector = mprojector.build_projector(config, config.input_projector_type)

    def forward(self, input):
        return self.projector(input)

class OutputProjector(MProjector):
    """输出投影器"""

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.projector = mprojector.build_projector(config, config.output_projector_type)

    def forward(self, input):
        return self.projector(input)


class MMUnderstander(nn.Module):
    """多模态理解器"""

    def __init__(self, config, *args):
        super().__init__()
        self.word_embbeding = nn.Module()
        self.args = args
        # 模态编码器
        self.m_encoder = MEncoder(config)
        # 输入投影器
        self.input_projector = InputProjector(config)
        self.backbone = self.getBackbone(config)

    def forward(self, **kwargs):
        """forward函数

        应该接受不同的输入形式, 例如文本, 图像, 音频等, 并返回对应的理解结果
        :param type: 输入的类型, 例如文本, 图像, 音频等, 见枚举类
        :param kwargs: 输入的数据, 从type中取值, 如果为文本, 则输入为语句, 其他的则用m_encoder

        TODO: 看下python的枚举类
        """
        input_embeds = self.embed(kwargs['type'], kwargs['input'])
        attention_mask = kwargs["attention_mask"]
        outputs = self.backbone(input_embeds=input_embeds,
                                attention_mask=attention_mask)
        return outputs

    def embed(self, type: str, input: Any) -> torch.Tensor:
        """按类型取得对应的embedding, 该embedding为backbone的输入

        Args:
            type (str): 输入的类型, 例如文本, 图像, 音频等, 见枚举类
            input (Any): 输入的数据, 如果是文本, 应为input_ids, 如果是图像, 应为图像的tensor, 如果是音频, 应为音频的tensor
        """
        # 搞个策略模式
        if (type == Modality.TEXT.value):
            return self.word_embbeding(input)
        # elif

        return torch.randn(1, 1)

    def getBackbone(self, config):
        """获取理解器的骨干模型, 基本为大模型"""
        return torch.load(config.model_path)



class MGenerator(torch.nn.Module):
    """多模态生成器"""
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = mgenerator.build_gengenerator(config)

    def forward(self, embedding):
        return self.generator(embedding)

class MMGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_projector = OutputProjector(config)
        self.generator = MGenerator(config)

    def forward(self, embedding):
        projection = self.output_projector(embedding)
        return self.generator(projection)




class ImagerGenerator(MGenerator):
    """图像生成器"""

    def __init__(self, config, *args):
        super().__init__(config)
        self.args = args

    def forward(self, sentences):
        pass


class MultiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_understander = MMUnderstander(config)
        self.mm_generator = MMGenerator(config)

    def forward(self, **kwargs):
        """forward函数

        应该接受不同的输入形式, 例如文本, 图像, 音频等, 并返回对应的理解结果
        :param type: 输入的类型, 例如文本, 图像, 音频等, 见枚举类
        :param kwargs: 输入的数据, 从type中取值, 如果为文本, 则输入为语句, 其他的则用m_encoder
        """
        input_embeds = self.mm_understander(**kwargs)
        outputs = self.mm_generator(input_embeds)
        return outputs