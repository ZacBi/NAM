
import re
from torch import nn

from transformers.activations import ACT2FN


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
    act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, config.input_projector_depth):
        modules.append(act_fn) # type: ignore
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

