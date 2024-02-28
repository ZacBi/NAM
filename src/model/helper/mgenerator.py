


import torch


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
    return torch.nn.Module(config, **kwargs)