import torch

ARCH_MAJOR, _ = torch.cuda.get_device_capability()

match ARCH_MAJOR:
    case 9:
        from pithtrain.operators.mla.tilelang import MLA
    case 10:
        from pithtrain.operators.mla.triton import MLA
    case _:
        from pithtrain.operators.mla.pytorch import MLA

__all__ = ["MLA"]
