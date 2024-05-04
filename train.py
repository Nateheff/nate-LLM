import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

from nate.model import Nate, Config
from nate.Tokenizer import Tokenizer
from helpers import get_batch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_size:int = 750 #how many rows we will use from the dataset in one training session. (One row is 
# ~100,000 characters so our dataset is roughly data_size * 100,000 characters
max_steps = 50000
check_interval = 1000

def train():
    
    m = Nate(Config)
    model = m.to(device)

    tok = Tokenizer()
    tok.load()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    data = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
    loader = DataLoader(data, batch_size=Config.batch_size)

    train = ""
    for i,row in enumerate(loader):
        for text in row['text']:
            train += text
        if(i == data_size):
            break
    
    full_tokens = tok.encode_ordinary(train)
    full_tokens = torch.tensor(full_tokens)
    for i in range(max_steps):
        x,y = get_batch(full_tokens, Config.batch_size, Config.max_context_length, device)
        logits,loss = model(x,y)
        if i % check_interval == 0:
            print(f"Step: {i} Train Loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                f="model.pt")


if __name__ == "__main__":  
    train()