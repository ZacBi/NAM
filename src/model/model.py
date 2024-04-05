#############
# MODEL ZOO #
#############

from cgitb import text
from typing import Any

import torch
from constant import Modality
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from torch import nn
from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoModelForPreTraining, AutoModelWithLMHead,
                          AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, RobertaModel)
from transformers.activations import ACT2FN

modelpath = "/data/MODELS/"


#############
# PROJECTOR #
#############


class IdentityProjector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_mlp_projector(config, **kwargs):
    """构建映射器， config.projector_type 为 mlp 时使用

    Args:
        config (_type_): _description_
        delay_load (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    # TODO: 异常处理
    act_fn = ACT2FN[config.hidden_act] if isinstance(
        config.hidden_act, str) else config.hidden_act

    modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, config.input_projector_depth):
        modules.append(act_fn)  # type: ignore
        modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    return nn.Sequential(*modules)


def build_projector(config, projector_type):

    if not projector_type:
        raise ValueError('projector_type is required')

    if projector_type == 'mlp':
        return build_mlp_projector(config)

    if projector_type == 'identity':
        return IdentityProjector()

    raise ValueError(f'Unknown projector type: {projector_type}')


class MProjector(nn.Module):
    """投影器, 用于将多模态理解器的输出投影到一个统一的空间中"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class InputProjector(MProjector):
    """输入投影器"""

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.projector = build_projector(
            config, config.input_projector_type)

    def forward(self, input):
        return self.projector(input)


class OutputProjector(MProjector):
    """输出投影器"""

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.projector = build_projector(
            config, config.output_projector_type)

    def forward(self, input):
        return self.projector(input)


#############
# GENERATOR #
#############

def build_gengenerator(config, **kwargs):
    """构建生成器

    Args:
        config (_type_): _description_
        delay_load (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # 图像类默认用stable diffusion, 从pipleline取
    if config.generator_type == "image":
        return SDImageGenerator(config)

    raise ValueError("不支持的生成器类型")


class MGenerator(nn.Module):
    """多模态生成器"""

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = build_gengenerator(config)

    def forward(self, embedding):
        return self.generator(embedding)


class ImageGenerator(MGenerator):
    """图像生成器, 默认用stable diffusion"""

    def __init__(self, config, *args):
        super().__init__(config)
        self.config = config
        self.args = args

    def forward(self, sentences):
        pass


class SDImageGenerator(ImageGenerator):
    r"""stable difussion 图像生成器"""

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            config.generator_model_pth, torch_dtype=torch.float16, variant="fp16")

    def forward(self, hidden_states):
        # 默认冻结状态
        outputs = self.pipeline(prompt_embeds=hidden_states)  # type: ignore
        return outputs


###########
# ENCODER #
###########

class MEncoder(torch.nn.Module):
    """模态编码器"""

    def __init__(self, args):
        super().__init__()
        self.args = args


class MMUnderstander(nn.Module):
    """多模态理解器"""

    def __init__(self, config):
        super().__init__()
        self.input_type = config.input_type
        # 模态编码器, 如果是非文本的话
        if self.input_type != Modality.TEXT.value:
            self.m_encoder = MEncoder(config)
            self.input_projector = InputProjector(config)
        self.backbone = self.getBackbone(config)

    def forward(self, inputs):
        """forward函数

        应该接受不同的输入形式, 例如文本, 图像, 音频等, 并返回对应的理解结果
        :param type: 输入的类型, 例如文本, 图像, 音频等, 见枚举类
        :param kwargs: 输入的数据, 从type中取值, 如果为文本, 则输入为语句, 其他的则用m_encoder

        TODO: 看下python的枚举类
        """
        input_embeds, attention_mask = self.embed(inputs)
        outputs = self.backbone(input_embeds=input_embeds,
                                attention_mask=attention_mask)
        return outputs

    def embed(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """按类型取得对应的embedding, 该embedding为backbone的输入

        Args:
            type (str): 输入的类型, 例如文本, 图像, 音频等, 见枚举类
            input (Any): 输入的数据, 如果是文本, 应为input_ids, 如果是图像, 应为图像的tensor, 如果是音频, 应为音频的tensor
        """
        # 搞个策略模式
        if (type == Modality.TEXT.value):
            return self.word_embbeding(input)
        # elif

        return (torch.randn(1, 1), torch.randn(1, 1))

    def getBackbone(self, config):
        """获取理解器的骨干模型, 基本为大模型"""
        return torch.load(config.model_path)


class MMGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_projector = OutputProjector(config)
        self.generator = MGenerator(config)

    def forward(self, hidden_states):
        projection = self.output_projector(hidden_states)
        return self.generator(projection)


class MMModel(nn.Module):
    r"""Multi-Modality model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_understander = MMUnderstander(config)
        self.mm_generator = MMGenerator(config)

    def forward(self, inputs):
        """forward函数

        应该接受不同的输入形式, 例如文本, 图像, 音频等, 并返回对应的理解结果
        :param type: 输入的类型, 例如文本, 图像, 音频等, 见枚举类
        :param kwargs: 输入的数据, 从type中取值, 如果为文本, 则输入为语句, 其他的则用m_encoder
        """
        input_embeds = self.mm_understander(inputs)
        outputs = self.mm_generator(input_embeds)
        return outputs

    def intermediate(self, n) -> nn.Module:
        if self.config.backbone_type == 'llama':
            # llama的层数由num_hidden_layers决定, 所有的backbone的层数都需要前置到config里面
            return self.mm_understander.backbone.layers[n].mlp
        return nn.Module()

    def _register_itermediate_forward_hook(self, fn):
        for layer_idx in self.config.num_hidden_layers:
            self.intermediate(layer_idx).register_forward_hook(fn)


class MMModelPrune(MMModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, inputs):
        """forward函数

        应该接受不同的输入形式, 例如文本, 图像, 音频等, 并返回对应的理解结果
        :param type: 输入的类型, 例如文本, 图像, 音频等, 见枚举类
        :param kwargs: 输入的数据, 从type中取值, 如果为文本, 则输入为语句, 其他的则用m_encoder
        """
        input_embeds = self.mm_understander(inputs)
        outputs = self.mm_generator(input_embeds)
        return outputs

    def intermediate(self, n) -> nn.Module:
        if self.config.backbone_type == 'llama':
            # llama的层数由num_hidden_layers决定, 所有的backbone的层数都需要前置到config里面
            return self.mm_understander.backbone.layers[n].mlp
        return nn.Module()

    def _register_itermediate_forward_hook(self, fn):
        for layer_idx in self.config.num_hidden_layers:
            self.intermediate(layer_idx).register_forward_hook(fn)

#####################
# FAST VERIFICATION #
#####################

class FastVerifyText2ImageMMModel(nn.Module):
    """取默认配置即可"""
    def __init__(self, config) -> None:
        super().__init__(config)
        # sd 原生自带CLIPTextModel, 不需要再引入bert, 可以替换为One-peace
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            config.generator_model_pth, torch_dtype=torch.float16, variant="fp16").to('cuda')

    def foward(self, sentence: str):
        outputs = self.pipeline(sentence)
        return outputs.images[0]
