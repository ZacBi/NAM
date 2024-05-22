import torch

from transformers import OPTForCausalLM



# model：In order to reduce the use of video memory, the model here uses the GILLModel_1 in the models to initialize the model
# weight：The pre-obtained raw_emb the weight of each element
# act：Subject activation value on each floor：subject here means IMG0 IMG1…… ，IMG7
def getWout(model):
    # Assuming you are using the first layer for simplicity
    Wout_t = model.lm.model.decoder.layers[0].fc2.weight.t()
    Wout = Wout_t.unsqueeze(0)
    for i in range(1, 32):
        Wout_t = model.lm.model.decoder.layers[i].fc2.weight.t()
        Wout = torch.cat((Wout, Wout_t.unsqueeze(0)))
    return Wout  # (32,16384,4096)
# act: (8, 32, 16384) 8:nums of token


def pic_attr(model, weight, act):
    # get each layer's Wout：（32，16384，4096）
    Wout = getWout(model)
    # (32, 16384, 4096)
    if torch.cuda.is_available():
        Wout = Wout.to(torch.device("cuda:0"))
        weight = weight.to(torch.device("cuda:0"))
        act = act.to(torch.device("cuda:0"))
    else:
        print("no device is avail")
    act[0] = (act[0]-torch.min(act[0]))/(torch.max(act[0]) - torch.min(act[0]))
    out = act[0].unsqueeze(-1)*Wout
    scores = out*weight[0]
    scores = torch.sum(scores, dim=2)
    scores_sum = scores.unsqueeze(0)
    for i in range(1, weight.shape[0]):   # (1,8)
        act[i] = (act[i]-torch.min(act[i])) / \
            (torch.max(act[i]) - torch.min(act[i]))
        out = act[i].unsqueeze(-1)*Wout  # (32, 16384, 4096)
        scores = out*weight[i]  # (32, 16384, 4096)
        scores = torch.sum(scores, dim=2)
        scores_sum = torch.cat((scores_sum, scores.unsqueeze(0)), dim=0)
    # scores_sum：（8，32，16384）
    return torch.sum(scores_sum, dim=0)


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
