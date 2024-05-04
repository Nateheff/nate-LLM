import json
import requests

from datasets import load_dataset
from torch.utils.data import DataLoader
from nate.model import Config
from nate.Tokenizer import Tokenizer


tok = Tokenizer()
tok.load()
encoded = tok.encode_ordinary("As a Computer Science professor who teaches Machine Learning, this is probably my most anticipated video ever. I regularly use your videos to brush up on/review ML concepts myself and recommend them to my students as study aids. You explain these concepts in the clear, straightforward way that I aspire to. Thank you!")
decoded = tok.decode(encoded)
print(encoded, decoded)
print(len(encoded), len(decoded))



