from typing import Generic, TypeVar
from torch.nn import Module

CfgT = TypeVar("CfgT")

# probably not needed but nicer to work with
class Decoder(Module, Generic[CfgT]):
    def __init__(self, cfg: CfgT, dataset_cfg):
        super().__init__()
