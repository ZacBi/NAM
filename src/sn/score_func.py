
import torch


class Text2ImageScore:
    """calc score from pixel to the every layer of backbone
    all loss should be backward to calc gradient before call function"""

    def __init__(self, config) -> None:
        self.config = config

    def text_2_image_gradient_score(self, model, backbone_hidden_states, final_outputs):
        """pixel gradient score

        : param loss: the loss of the model, scalar
        : backbone_hidden_states: the hidden states of each mlp layer in the backbone, (batch_size, seq_len, hidden_size)
        : final_outputs: (batch_size, num_channels, height, width)

        参考 https://arxiv.org/pdf/2308.01544.pdf
        """
        score = {}
        if self.config.backbone_type == 'llama':
            # llama的outputs中的hidden_states 即为每一层的 mlp的输出可以直接使用
            # 直接取每一层的gradient * hiden_states
            for layer_idx in self.config.num_hidden_layers:
                # 重置梯度
                model.zero_grad()
                score[layer_idx] = torch.matmul(backbone_hidden_states[layer_idx], torch.autograd.grad(final_outputs, backbone_hidden_states[layer_idx])[0])
        return score

    def text_2_image_activation_score(self, loss, model):
        """pixel activattion score"""
        pass

    def text_2_iamge_integral_score(self, loss, model):
        """pixel integral score"""
        pass
