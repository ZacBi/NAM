#############
# MODEL ZOO #
#############

from typing import Any
from constant import Modality
from re import L
from token import STRING
from turtle import forward
from httpx import get
import torch
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer, AdapterConfig, AutoModelWithLMHead, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForPreTraining, AutoModelForCausalLM

modelpath = "/data/MODELS/"

# Basemodel


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.getmodel()
        self.args = args
        self.prompt_type = None

    def getmodel(self):
        pass

    def forward(self, sentences):
        pass

    def intermediate(self, n):
        pass

    def embed(self, input_ids):
        pass

    # NOTE: typo,  Verbalizer, 空间映射器
    def getverablizer(self):
        """ 
        空间映射器，顶层模型继承时需要重写
        """
        args = self.args
        print(args.verb)
        if (len(args.verb) == 0 or args.verb[0] == ''):
            # 下面huanjige
            positive = self.tokenizer.encode("positive")[1]
            negative = self.tokenizer.encode("negative")[1]
            neutral = self.tokenizer.encode("neutral")[1]
            conflict = self.tokenizer.encode("conflict")[1]
            if (self.args.num_labels == 2):
                # 标签在encoding中的位置
                self.pos = [negative, positive]
            if (self.args.num_labels == 3):
                self.pos = [negative, neutral, positive]
            if (self.args.num_labels == 4):
                self.pos = [conflict, negative, neutral, positive]
        elif (len(args.verb) == 1):
            self.pos = random.sample(list(range(50265)), self.num_labels)
        else:
            self.pos = [self.tokenizer.encode(word)[1] for word in args.verb]
        print(self.pos)
        print(len(self.pos))

    def processoutput(self, output):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize_parameters(self):
        pass

    # TODO: 扰动实验，用来证实skill neuron的重要性
    def addmask(self, thspath, lowlayer=0, highlayer=12, type="mean"):
        if (type == "mean"):
            self.bias = torch.tensor(torch.load(thspath)).cuda().mean(axis=1)
            # from IPython import embed;embed()
        elif (type == "zero"):
            self.bias = [0]*12
        if (type != "gaussian"):
            def save_std_outputs1_hook(k):
                def fn(_, __, output):
                    cmask = self.pmask[k]
                    bias = self.bias[k]
                    bias = bias*cmask
                    # from IPython import embed;embed()
                    output = output*(~cmask)
                    output += bias
                    return output
                return fn
            for k in range(lowlayer, highlayer):
                self.intermediate(k).register_forward_hook(
                    save_std_outputs1_hook(k))
        else:
            def save_std_outputs1_hook(k):
                def fn(_, __, output):
                    cmask = self.pmask[k]
                    bias = torch.randn(
                        [output.shape[0], 3072]).cuda()*self.args.alpha
                    bias = bias*cmask
                    output += bias.unsqueeze(dim=1)
                    return output
                return fn
            for k in range(lowlayer, highlayer):
                self.intermediate(k).register_forward_hook(
                    save_std_outputs1_hook(k))

class PromptBaseModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def save(self, path):
        parameter = {}
        state = self.state_dict()
        parameter["prompt"] = state["prompt"]
        parameter["pos"] = self.pos
        torch.save(parameter, path + "-backbone")

    def load(self, path):
        if (self.args.load_backbone):
            print("loading backbone from "+self.args.load_backbone)
            self.backbone.load_state_dict(torch.load(self.args.load_backbone))
        if (self.args.from_pretrained):
            parameter = torch.load(path + "-backbone")
            state = self.state_dict()
            state["prompt"] = parameter["prompt"]
            self.pos = parameter["pos"]
            self.load_state_dict(state)

    def optimize_parameters(self):
        return [{'params': [p for n, p in self.named_parameters() if "prompt" in n], 'weight_decay': 0.0}]


class LlavaPrompt(PromptBaseModel):
    def __init__(self, args):
        super().__init__(args)

    def getmodel(self):
        self.backbone = AutoModelForCausalLM.from_pretrained(
            modelpath + 'llava')

    def processoutput(self, outputs):
        return outputs.logits[:, 0, self.pos].squeeze(dim=1)

    def intermediate(self, n):
        return self.backbone.roberta.encoder.layer[n].intermediate

    def embed(self, input_ids):
        return self.backbone.roberta.embeddings.word_embeddings(input_ids).detach()


class LlavaPrunePrompt(PromptBaseModel):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelpath+"llava")
        self.layer_num = 12
        self.layer_width = 3072
        super().__init__(args)

    def getmodel(self):
        self.backbone = torch.load("prune_structure/PruneLlava")

    def processoutput(self, outputs):
        return outputs.logits[:, 0, self.pos].squeeze(dim=1)

    def intermediate(self, n):
        return self.backbone.roberta.encoder.layer[n].intermediate

    def embed(self, input_ids):
        return self.backbone.roberta.embeddings.word_embeddings(input_ids).detach()

    def load(self, path):
        parameter = torch.load(path + "-backbone")
        self.pos = parameter["pos"]
        self.load_state_dict(parameter, strict=False)

    def save(self, path):
        parameter = self.state_dict()
        parameter["pos"] = self.pos
        torch.save(parameter, path + "-backbone")





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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class OutputProjector(MProjector):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



class MMUnderstander(torch.nn.Module):
    """多模态理解器"""

    def __init__(self, args):
        super().__init__()
        self.word_embbeding = torch.nn.Module()
        self.args = args
        # 模态编码器
        self.m_encoder = MEncoder(args)
        # 输入投影器
        self.input_projector = InputProjector(args)
        self.backbone = torch.nn.Module()

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

    def getBackbone(self):
        """获取理解器的骨干模型, 基本为大模型"""
        pass

class MGenerator(torch.nn.Module):
    """多模态生成器"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_projector = OutputProjector(args)
        self.generator = torch.nn.Module()

    def forward(self, embedding):
        projection = self.output_projector(embedding)



class ImagerGenerator(MGenerator):
    """图像生成器"""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, sentences):
        pass
