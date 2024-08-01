from model.model import Config
from dataclasses import dataclass

@dataclass
class LoraConfig:
    rank:int = 4
    head_dim:int = Config.head_dim
