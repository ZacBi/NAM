import torch

from transformers import OPTForCausalLM


# model：In order to reduce the use of video memory, the model here uses the GILLModel_1 in the models to initialize the model
# weight：The pre-obtained raw_emb the weight of each element
# act：Subject activation value on each floor：subject here means IMG0 IMG1…… ，IMG7
def getWout(model):
    """
    获取模型所有解码器层的输出权重矩阵。

    参数:
    - model: 模型对象，其中包含了要获取权重的解码器层。

    返回值:
    - Wout: 所有解码器层的输出权重组成的张量，形状为（层数，输入特征数，输出特征数）。
    """

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


def calculate_scores(Wout, weights, acts):
    """Calculate and sum up the scores."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Wout = Wout.to(device)
    weights = weights.to(device)
    acts = acts.to(device)

    min_acts = torch.min(acts, dim=-1).values
    max_acts = torch.max(acts, dim=-1).values
    norm_acts = (acts - min_acts.unsqueeze(-1).expand_as(acts)) / \
        (max_acts - min_acts).unsqueeze(-1).expand_as(acts)

    score = torch.sum(norm_acts * Wout * weights, dim=-1)
    scores_sum = score.sum(dim=0)
    # scores_sum：(8，32，16384)
    return scores_sum

def pic_attr(model, weight, act):
    Wout = getWout(model)
    result = calculate_scores(Wout, weight, act)
    return result


if __name__ == "__main__":
    # 1. load backbone
    llm_backbone_path = "facebook/opt-6.7b"
    llm_backbone = OPTForCausalLM.from_pretrained(
        pretrained_model_name_or_path=llm_backbone_path)

    # 2. after call src/gill/generated_image_interpret.ipynb you'll get the attribution for raw embeds
    attribution_path = 'path_to_the_attribution'
    attribution = torch.load(attribution_path)

    # 3. activations of each ffn layers in llm_model
    act_path = 'path_to_the_activations'
    act = torch.load(act_path)

    # 4. calculate the attribution of all neurons to the generated image
    neuron_attribution = pic_attr(llm_backbone, attribution, act)

    torch.save(neuron_attribution, 'path_to_the_neuron_attribution')
