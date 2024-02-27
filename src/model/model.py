#############
# MODEL ZOO #
#############

from re import L
from turtle import forward
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
