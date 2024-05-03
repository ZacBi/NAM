# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch InstructBLIP model."""
import sys
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig
from transformers import LlamaForCausalLM

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/instructblip-flan-t5-xl"

INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/instructblip-flan-t5-xl",
    # See all InstructBLIP models at https://huggingface.co/models?filter=instructblip
]


@dataclass
# 从transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutputcopy过来的，把blip2替换成instructBlip
# Copied from transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput with Blip2->InstructBlip
class InstructBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """
# Optional，表示该变量可以是指定类型的数据，也可以是 None。在这里，指定类型为 Tuple[torch.FloatTensor]，即一个由 torch.FloatTensor 类型组成的元组
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    # 返回类型为 Tuple[Any]，表示返回一个由任意类型数据组成的元组。
    # self 是一个对象实例，通常是一个类的实例，而 k 则是一个键（key），返回self实例中所有的值（元组形式）
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            # getattr用于获取对象的属性值。
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.blip.modeling_blip.BlipVisionEmbeddings with Blip->InstructBlip
# visionembedding
# InstructBlipVisionConfig这个应该是自己定义的一种配置变量类
class InstructBlipVisionEmbeddings(nn.Module):
    def __init__(self, config: InstructBlipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        # 随机初始化了一个模型参数
        # class_embedding 是一个类成员变量（属性），通常用来表示模型中的类别嵌入
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # patch_embedding 通常用来表示图像中的图像块（patches）嵌入的操作或模块
        # self.patch_embedding 是一个 nn.Conv2d 的实例，它被用来实现图像块的转换操作，将输入的图像分成小的图像块并将其映射到嵌入空间中。
        # out_channels=self.embed_dim。输出的通道数也就是嵌入维度
        # kernel_size和stride大小相同，确保不重叠的提取图像块
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        # self.num_patches通常用来表示图像被分成的图像块数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 这个变量通常用来表示位置编码中的位置数量，图像块数量加上一个额外的位置用于表示全局位置
        self.num_positions = self.num_patches + 1
        # self.num_positions个位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))
    # 像素值（应该是有好多张图片作为batch）作为输入，返回torch.Tensor类型
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        # 获取了 self.patch_embedding 层的权重张量的数据类型
        target_dtype = self.patch_embedding.weight.dtype
        # 这行代码将输入的 pixel_values 通过 self.patch_embedding 层进行处理，得到图像块的嵌入表示。
        # 处理后的 patch_embeds 张量的形状为 [*, width, grid, grid]，其中 * 表示其他维度，width 是宽度，grid 是网格大小。
        # 这里我感觉是图像块是grid*grid编码，width就是分割成多少图像块，一个picture
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        # 展平为[*, width, grid*grid]，transpose 操作将第1维和第2维交换位置[*, grid*grid， width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # self.class_embedding.expand(batch_size, 1, -1) [batch_size, 1, -1]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # 这行代码将类别嵌入和图像块嵌入拼接在一起，形成最终的嵌入表示 embeddings，在第1维度上进行拼接。
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 这行代码将位置编码与之前的嵌入表示相加，以获得最终的位置编码嵌入。位置编码已经在前面定义好，通过索引和类型转换操作来与嵌入表示相加
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings


# Copied from transformers.models.blip_2.modeling_blip_2.Blip2Attention with Blip2->InstructBlip
class InstructBlipAttention(nn.Module):
    # 多头注意力机制
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 要求整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放注意力机制中的点积注意力矩阵
        self.scale = self.head_dim**-0.5
        # 池化层
        self.dropout = nn.Dropout(config.attention_dropout)

        # small tweak here compared to CLIP, no bias here
        # 定义了一个线性变换层 self.qkv，输入维度为 self.embed_dim，输出维度为 3 * self.embed_dim，并且不使用偏置（bias=False）
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        # 根据配置中 config.qkv_bias 的设置，来决定是否为查询（Q）和值（V）添加偏置
        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None
        #
        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)
        # 最后这个线性层是什么
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 先调用 view 方法对张量进行重塑操作，将其变换为形状为 (batch_size, seq_len, num_heads, head_dim)
        # transpose (batch_size, num_heads, seq_len, head_dim)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # bsz: 表示 batch size，即批次大小，表示输入数据中样本的数量。
        # tgt_len: 表示 target length，即目标长度，通常用来表示序列的长度。
        # embed_dim: 表示 embedding dimension，即嵌入维度，通常用来表示词嵌入或者特征的维度。
        # hidden_states 可能表示模型中某一层的隐藏状态或者输出
        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)
        # 这里的参数 (2, 0, 3, 1, 4) 指定了新的维度排列顺
        # 最后一个维度是head_dim,表示每个注意力头的维度
        # batch_size 是批次大小，即样本数量。
        # tgt_len 是目标长度，表示序列的长度。
        # 3 表示有三个部分，分别对应查询（Q）、键（K）和值（V）。
        # num_heads 是多头注意力机制中的头数。
        # head_dim 是每个注意力头的维度。
        # 将第二维移到最前面，也就是3：表示Q,K,V
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        # 第一个维度的三个向量矩阵分别是固定的q,k,v经过Q,K,V后
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        # 对注意力矩阵进行掩码操作
        # 存在一个名为 head_mask 的掩码张量
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)
        # 将 context_layer 的最后两个维度（通常是序列长度和隐藏单元数）替换为 self.embed_dim，以便后续投影操作。
        # 最后两个维度一般是注意力头数和每个头的维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        # reshape得到新形状
        context_layer = context_layer.reshape(new_context_layer_shape)
        # 多头注意力
        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# Copied from transformers.models.blip.modeling_blip.BlipMLP
class InstructBlipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 从预定义的激活函数字典 ACT2FN 中获取配置中指定的激活函数，并保存在 self.activation_fn 中
        self.activation_fn = ACT2FN[config.hidden_act]
        # 两个线性层
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.blip.modeling_blip.BlipEncoderLayer with Blip->InstructBlip
class InstructBlipEncoderLayer(nn.Module):
    def __init__(self, config: InstructBlipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # 多头注意力层
        self.self_attn = InstructBlipAttention(config)
        # 归一化层--MLP--归一化层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = InstructBlipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存变量以便后续进行残差连接
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class InstructBlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    一个处理权重初始化的抽象类和用于下载和加载预训练模型的简单接口
    """

    config_class = InstructBlipConfig
    # 基础模型前缀
    base_model_prefix = "blip"
    # 模型是否支持梯度检查点（gradient checkpointing）技术
    supports_gradient_checkpointing = True
    # 一组模块的名称
    _no_split_modules = ["InstructBlipAttention", "InstructBlipQFormerMultiHeadAttention"]
    _keep_in_fp32_modules = []

    # Copied from transformers.models.blip_2.modeling_blip_2.Blip2PreTrainedModel._init_weights with Blip2->InstructBlip
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, InstructBlipVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InstructBlipEncoder):
            module.gradient_checkpointing = value


INSTRUCTBLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InstructBlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

INSTRUCTBLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`InstructBlipProcessor`]. See
            [`InstructBlipProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

INSTRUCTBLIP_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`InstructBlipProcessor`]. See
            [`InstructBlipProcessor.__call__`] for details.

        qformer_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the Q-Former. Input tokens can optionally be provided
            to serve as text prompt, which the Q-Former model will encode.

            Indices can be obtained using [`InstructBlipProcessor`]. See [`InstructBlipProcessor.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)

        qformer_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.

            Indices can be obtained using [`InstructBlipProcessor`]. See [`InstructBlipProcessor.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)

        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            Only relevant in case an encoder-decoder language model (like T5) is used.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.blip.modeling_blip.BlipEncoder with Blip->InstructBlip
class InstructBlipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InstructBlipEncoderLayer`].

    Args:
        config (`InstructBlipConfig`):
            The corresponding vision configuration for the `InstructBlipEncoder`.
    """

    def __init__(self, config: InstructBlipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([InstructBlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from transformers.models.blip.modeling_blip.BlipVisionModel with Blip->InstructBlip, BLIP->INSTRUCTBLIP
class InstructBlipVisionModel(InstructBlipPreTrainedModel):
    main_input_name = "pixel_values"
    config_class = InstructBlipVisionConfig

    def __init__(self, config: InstructBlipVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = InstructBlipVisionEmbeddings(config)
        self.encoder = InstructBlipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=InstructBlipVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


# Copied from transformers.models.blip_2.modeling_blip_2.Blip2QFormerMultiHeadAttention with Blip2->InstructBlip
class InstructBlipQFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        # 注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 每个头的注意力维度
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 难道不等于hidden_size吗？ 应该是相等的，因为上面已经满足可以整除
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 这就是查询向量learnable_query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # config.attention_probs_dropout_prob丢弃概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # getattr(config, "position_embedding_type", "absolute")会确认是否存在该属性，然后如果需要的话返回属性对应的值
        # 从 config 对象中获取 position_embedding_type 属性的值，如果该属性不存在，则使用默认值 "absolute"。
        # 这个属性用于指定位置编码（Positional Embedding）的类型，可以是 "absolute"、"relative_key" 或 "relative_key_query"。
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 第一个参数 2 * config.max_position_embeddings - 1：表示嵌入层要处理的不同索引的数量，即嵌入的维度大小。在这里，计算出的数量为 2 * config.max_position_embeddings - 1，即索引范围为从 0 到 (2 * config.max_position_embeddings - 2)。
            # 第二个参数 self.attention_head_size：表示每个嵌入向量的维度大小，即每个索引会映射为一个长度为 self.attention_head_size 的向量。
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False
    # 用于保存传入的注意力梯度（attention gradients）到实例的属性 self.attn_gradients 中
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
    # 返回保存在实例属性 self.attn_gradients 中的注意力梯度
    def get_attn_gradients(self):
        return self.attn_gradients
    # 用于保存传入的注意力图（attention map）到实例的属性 self.attention_map 中
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
    # 返回保存在实例属性 self.attention_map 中的注意力图
    def get_attention_map(self):
        return self.attention_map
    # 用于将输入张量进行形状变换，以便适应注意力头的形状。在这个方法中，张量 x 被重新排列为一个新的形状，然后执行维度置换操作以返回变换后的张量
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    # 表示模型的输入隐藏状态（hidden states），通常是待处理的序列数据或特征表示
    # 用于指定哪些位置需要被掩盖的注意力掩码（attention mask）。掩盖的位置将不被模型考虑。
    # 用于指定哪些注意力头应该被屏蔽的头掩码（head mask），以便控制注意力头的作用
    # 编码器的隐藏状态（encoder hidden states），通常用于交叉注意力（cross-attention）任务中，作为键和值的来源。
    # 编码器的注意力掩码（encoder attention mask），用于指定编码器中哪些位置需要被掩盖。
    # 用于存储过去的键值对，以便支持自回归模型的解码。
    # 一个布尔值，表示是否输出注意力权重。如果设置为True，函数将返回注意力权重的信息。
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # 如果这个被实例化为交叉注意力模块，那么键（keys）和值（values）来自一个编码器；注意力掩码（attention mask）需要使编码器的填充标记不被关注
        # 非空，置为true，表示这个对象将被用作交叉注意力模块。
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            #
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 这里面的layer其实是Q,K,V经过矩阵Q,K,V处理后得到的结果
        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)
        # 计算Q*K的转置
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        ##### 计算相对位置编码（relative positional embedding）对注意力得分（attention scores）的影响
        # 通过检查 self.position_embedding_type 的值是否为 "relative_key" 或 "relative_key_query" 来确定是否需要计算相对位置编码
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 第一维（索引 0）是 batch_size，表示一个批次中样本的数量。
            # 第二维（索引 1）是 sequence_length，表示序列的长度，即时间步数或序列中的元素数量。
            # 第三维（索引 2）是 hidden_size，表示每个时间步或元素的隐藏状态向量的维度大小。
            # 计算相对位置编码！！！ position_ids_l表示左侧，position_ids_r表示右侧
            seq_length = hidden_states.size()[1]
            # 列向量：（1，seq_length-1)，用于表示左侧位置地ID
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            # 行向量: (1, seq_length-1)， 用于表示右侧位置ID
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            # 广播机制：得到[seq_length-1, seq_length-1]的向量
            # 例如三维的distance，最后得到是这样的结果
            # 0 -1 -2
            # 1  0 -1
            # 2  1  0
            distance = position_ids_l - position_ids_r
            # 使用位置编码模型（self.distance_embedding）计算相对位置编码张量 positional_embedding，并将其转换为和 query_layer 张量相同的数据类型。
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
            # 根据 self.position_embedding_type 的值，分别计算相对位置得分（relative_position_scores）
            # 仅使用查询张量 query_layer 和位置编码张量计算得分，并将得分添加到注意力得分中
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            # self.position_embedding_type 是 "relative_key_query"，则同时使用查询张量和键张量与位置编码张量计算得分，并将这两个得分添加到注意力得分中
            '''在注意力机制中引入相对位置，具体数学原理不知，反正可以使模型在计算注意力权重的时候考虑相对位置关系'''
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # 归一化操作
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            # 保存注意力权重
            self.save_attention_map(attention_probs)
            # 将 save_attn_gradients 方法注册为 attention_probs 的梯度钩子。这可以用来在反向传播过程中保存注意力权重的梯度信息，以便进行梯度分析或其他操作
            # 度钩子允许用户在张量的梯度被计算时执行额外的操作，比如记录梯度、修改梯度或者梯度可视化：
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 将一些token置为0
        attention_probs_dropped = self.dropout(attention_probs)

        # 主要用于计算上下文向量 context_layer
        # Mask heads if we want to
        if head_mask is not None:
            # 如果提供了头注意力掩码，则将注意力权重 attention_probs_dropped 与头注意力掩码相乘，实现对特定注意力头的屏蔽操作。
            # attention_probs_dropped是什么？ 注意力权重（attention_probs）表示了每个查询向量对所有键值对的注意力分配，这里的查询向量是可学习的query_tokens,键值对是注意力机制中的value矩阵和key矩阵
            attention_probs_dropped = attention_probs_dropped * head_mask
        # 根据注意力权重 attention_probs_dropped 和值矩阵 value_layer 计算上下文向量
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        # 对计算得到的上下文向量进行维度置换，以便后续的形状变换操作
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 根据上下文向量的维度计算新的形状，用于进行下一步的形状变换
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 根据新的形状对上下文向量进行形状变换，将其重新排列为期望的形状。
        context_layer = context_layer.view(*new_context_layer_shape)
        # 根据新的形状对上下文向量进行形状变换，将其重新排列为期望的形状。
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # 将过去的键值对信息 past_key_value 添加到输出元组中。
        outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->InstructBlipQFormer
class InstructBlipQFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.blip_2.modeling_blip_2.Blip2QFormerAttention with Blip2->InstructBlip
class InstructBlipQFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention = InstructBlipQFormerMultiHeadAttention(config, is_cross_attention)
        self.output = InstructBlipQFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->InstructBlipQFormer
class InstructBlipQFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->InstructBlipQFormer
class InstructBlipQFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class InstructBlipQFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = InstructBlipQFormerAttention(config)

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = InstructBlipQFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = InstructBlipQFormerIntermediate(config)
        self.output = InstructBlipQFormerOutput(config)

        self.intermediate_query = InstructBlipQFormerIntermediate(config)
        self.output_query = InstructBlipQFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.blip_2.modeling_blip_2.Blip2QFormerEncoder with Blip2->InstructBlip
class InstructBlipQFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [InstructBlipQFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class InstructBlipQFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids.to(embeddings.device))
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class InstructBlipQFormerModel(InstructBlipPreTrainedModel):
    """
    Querying Transformer (Q-Former), used in InstructBLIP. Slightly modified from BLIP-2 as it also takes the
    instruction as input.
    """

    def __init__(self, config: InstructBlipQFormerConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InstructBlipQFormerEmbeddings(config)

        self.encoder = InstructBlipQFormerEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})",
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        query_embeds=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and query_embeds is None:
            raise ValueError("You have to specify query_embeds when input_ids is None")

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        )

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision
    encoder, Querying Transformer (Q-Former) and a language model.

    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.
    InstructBLIP 模型用于根据图像和可选文本提示生成文本。该模型包括一个视觉编码器、一个查询变换器（Q-Former）和一个语言模型。
    用户可以选择性地向模型传递 input_ids，作为文本提示，以便让语言模型继续提示的文本。否则，语言模型将从 [BOS]（序列开始）标记开始生成文本。
    """,
    INSTRUCTBLIP_START_DOCSTRING,
)
class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        self.vision_model = InstructBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        if config.use_decoder_only_language_model:
            language_model = LlamaForCausalLM(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

        self.cal_score = False
        self.cal_activations = False
        self.cal_new_score = False
        self.shuffle = False
        self.idx = None
        self.start = None
        self.end = None
        self.scores = []
        self.new_scores = []
        self.hidden = None
        self.act = None
        self.cal_activations = False
        self.config = config.text_config

    def get_model(self):
        return self.model
    # 用于设置获取得分的相关参数
    def set_get_scores(self, start, end):
        self.scores = []
        self.cal_score = True
        self.start = start
        self.end = end
    # 用于设置获取新得分的相关参数
    def set_get_new_scores(self, start, end):
        self.new_scores = []
        self.cal_new_score = True
        self.start = start
        self.end = end
    # 用于设置模型获取激活值的相关参数
    # start 和 end 是作为参数传递给这个函数的起始和结束位置的值
    def set_get_activations(self, start, end):
        # 将模型的激活值 self.activations 和 self.last_activation 设置为初始状态，即置为 None 和空列表 []
        self.activations = None
        self.last_activation = []
        # 将计算激活值的标志 self.cal_activations 设置为 True，表示需要计算激活值。
        self.cal_activations = True
        # 将计算激活值的标志 self.cal_activations 设置为 True，表示需要计算激活值。
        self.start = start
        self.end = end

    def get_activations(self):
        return self.last_activation

    def get_scores(self):
        return self.scores

    def get_new_scores(self):
        return self.new_scores

    def set_shuffle(self):
        self.shuffle = True

    def set_idx(self, idx):
        self.idx = idx
    # 返回语言模型的输入嵌入
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
    
    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        pass
        # if not self.config.use_decoder_only_language_model:
        #     self.language_model.encoder.embed_tokens = self.language_model.shared
        #     self.language_model.decoder.embed_tokens = self.language_model.shared
    # 这个函数包含了一些预处理的操作，以使模型兼容 accelerate 加速库
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + InstructBLIP + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=InstructBlipForConditionalGenerationModelOutput, config_class=InstructBlipVisionConfig
    )
    def forward(
        self,
        # pixel_values: 图像的像素值，类型为 torch.FloatTensor
        pixel_values: torch.FloatTensor = None,
        # qformer_input_ids: Q-Former 的输入标记，类型为 torch.FloatTensor
        qformer_input_ids: torch.FloatTensor = None,
        # 可选的 Q-Former 注意力掩码，类型为 torch.LongTensor
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        # 过去的键值
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否使用缓存，类型为 bool
        use_cache: Optional[bool] = None,
        # 输入标记
        input_ids: Optional[torch.FloatTensor] = None,
        # 注意力掩码
        attention_mask: Optional[torch.LongTensor] = None,
        # 是否输出注意力
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        #  用于计算语言模型损失的标签
        labels: Optional[torch.LongTensor] = None,
        # 是否返回字典形式的输出
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(pixel_values)
        # print(qformer_input_ids)
        # print(qformer_attention_mask)

        # 这段代码是模型前向传播中的第一步，通过视觉编码器（vision_model）将图像数据进行处理，以获取图像嵌入（image embeddings）
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)


        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.shuffle:
            idx = torch.cat([self.idx, torch.tensor(range(32, inputs_embeds.shape[1]))], dim=0)
            inputs_embeds = inputs_embeds[:, idx]
        # 每个空列表都代表一个隐藏层，检查是否需要记录激活值。
        if self.cal_activations:
            self.act = [[] for _ in range(self.config.num_hidden_layers)]
            
            def forward_hook(n):
                def fn(_, input, output):
                    self.act[n].append(output.detach())

                return fn
            # 这里的这种模型架构是需要自己去写吗，为每一层注册hook
            handle_act = [self.language_model.model.layers[n].mlp.act_fn.register_forward_hook(forward_hook(n)) for n in
                          range(self.config.num_hidden_layers)]


        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        logits = outputs.logits


        if self.cal_activations:
            # 获取每个隐藏层的最后一个激活值
            last_activation = [(self.act[n][0].detach()[0, -1, :]).half().cpu() for n in
                               range(self.config.num_hidden_layers)]
            self.last_activation.append(last_activation)
            for h in handle_act:
                del h

        return outputs
# 准备模型生成（generation）时的输入参数
    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 方法接受多个参数：
        # 包括输入的 input_ids、过去的键值对（past_key_values，默认为 None）、注意力掩码（attention_mask，默认为 None）、输入的嵌入（inputs_embeds，默认为 None）以及其他关键字参数 **kwargs
        # if self.cal_score or self.cal_activations:
        past_key_values = None
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": kwargs.get("pixel_values", None),
                "qformer_input_ids": kwargs.get("qformer_input_ids", None),
                "qformer_attention_mask": kwargs.get("qformer_attention_mask", None),
            }
        )
        return model_inputs

    # @torch.no_grad()
    # def generate(
    #     self,
    #     pixel_values: torch.FloatTensor,
    #     qformer_input_ids: Optional[torch.LongTensor] = None,
    #     qformer_attention_mask: Optional[torch.LongTensor] = None,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.LongTensor] = None,
    #     **generate_kwargs,
    # ) -> torch.LongTensor:
    #     """
    #     Overrides `generate` function to be able to use the model as a conditional generator.
    #
    #     Args:
    #         pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
    #             Input images to be processed.
    #         qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
    #             The sequence used as a prompt to be fed to the Q-Former module.
    #         qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
    #             Mask to avoid performing attention on padding token indices.
    #         input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
    #             The sequence used as a prompt for the generation.
    #         attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
    #             Mask to avoid performing attention on padding token indices.
    #
    #     Returns:
    #         captions (list): A list of strings of length batch_size * num_captions.
    #     """
    #     if hasattr(self, "hf_device_map"):
    #         # preprocess for `accelerate`
    #         self._preprocess_accelerate()
    #
    #     batch_size = pixel_values.shape[0]
    #     image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
    #
    #     image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
    #
    #     query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
    #     if qformer_attention_mask is None:
    #         qformer_attention_mask = torch.ones_like(qformer_input_ids)
    #     qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
    #     query_outputs = self.qformer(
    #         input_ids=qformer_input_ids,
    #         attention_mask=qformer_attention_mask,
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_attention_mask,
    #         return_dict=True,
    #     )
    #     query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
    #
    #     language_model_inputs = self.language_projection(query_output)
    #     language_attention_mask = torch.ones(
    #         language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    #     )
    #
    #     if input_ids is None:
    #         input_ids = (
    #             torch.LongTensor([[self.config.text_config.bos_token_id]])
    #             .repeat(batch_size, 1)
    #             .to(image_embeds.device)
    #         )
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids)
    #     attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)
    #
    #     print(input_ids)
    #
    #     # concatenate query embeddings with prompt embeddings
    #     inputs_embeds = self.get_input_embeddings()(input_ids)
    #     inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
    #
    #     generate_kwargs.pop('pixel')
    #
    #
    #     outputs = self.language_model.generate(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         **generate_kwargs,
    #     )
    #
    #
    #     # print("outputs, ", outputs)
    #
    #     # the InstructBLIP authors used inconsistent tokenizer/model files during training,
    #     # with the tokenizer's bos token being set to </s> which has ID=2,
    #     # whereas the model's text config has bos token id = 0
    #     if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
    #         outputs[outputs == 0] = 2
    #
    #     return outputs
