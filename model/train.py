import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from model import Nate, Config
from Tokenizer import Tokenizer
from helpers import get_batch, create_targets, pad
from SAM.data import create_tok_dataset, create_dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_size:int = 750 #how many rows we will use from the dataset in one training session. (One row is 
# ~100,000 characters so our dataset is roughly data_size * 100,000 characters
max_steps = 50000
check_interval = 1000
max_epochs = 10

# data = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
# loader = DataLoader(data, batch_size=Config.batch_size)
# loader=None

train_tokens = " "
# for i, row in enumerate(loader):
#     for text in row['text']:
#         train_tokens += text
#     if i > 100:
#         break

def train_new_sam_tok():
    print("Training SAM tokenizer from scratch")
    tok = Tokenizer()
    tok.train(train_tokens, Config.vocab_size, loaded=False)
    tok.save("SAM/model_base.pickle")
    print("trained first")

def train_sam_tok():
    print("Training SAM Tokenizer")
    tok = Tokenizer()
    tok.load("SAM/model.pickle")

    tok.train(train_tokens, Config.vocab_size + 13, True)
    tok.save("SAM/model.pickle")
    print(tok.vocab)
    print("trained SAM tokenizer")


def train_tok():
    print("Training Tokenizer")
    tok = Tokenizer()
    tok.train(train, Config.vocab_size)
    tok.save()
    print("trained tokenizer")

def train():
    
    m = Nate(Config)
    model = m.to(device)

    tok = Tokenizer()
    tok.load()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train:str = ""

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

def train_sam():
    m = Nate(Config)
    model = m.to(device=device)
    
    tok = Tokenizer()
    tok.load("SAM/model.pickle")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    d_set = create_dataset()
    loader = DataLoader(dataset=d_set, shuffle=True, batch_size=2)
    print("Loop Start")
    for i in range(max_epochs):
        for x,y in loader:

            y = create_targets(x,y)
            x,y = tok.encode_many(x), tok.encode_many(y)

            y = [tokens[1:] for tokens in y]
            pad(x,y,Config.max_context_length, 0)
            # print(f"X: {x} {len(x[0])} \n Y: {y} {len(y[0])}")
            x,y = torch.tensor(x), torch.tensor(y)
            
            x,y = x.to(device), y.to(device)
            logits, loss = model(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {i} Loss: {loss}")

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                f="SAM/model.pt")
    print("Trained")
    


def test_sam():
    model = Nate(Config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = torch.load("SAM/model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    model = model.to(device)

    tok = Tokenizer()
    tok.load("SAM/model.pickle")

    test = "<|begin|> sam stop firing <|end|>"
    print("begin inference")
    toks = tok.encode_ordinary(test)
    toks = torch.tensor(toks)
    toks = toks.unsqueeze(0)
    toks = toks.to(device)

    logits = model.generate(toks,1)

    print(logits)
    
    print(tok.decode(logits.tolist()[0]))
    
specials_list = [" [WEATHER]", " [TIME]", " [TURRET]", " [MUSIC]", " [DATE]", " hey", " sam", " shoot", " fire", " intruder", " weather", " time", " what", " how", " red", " alert", " spotify", " music", " play", " song", " time", " what's", " how's"]


if __name__ == "__main__":
    # train_sam_tok()
    # train_new_sam_tok()
    # tok = Tokenizer()
    # tok.load("SAM/model.pickle")
    # print(tok.vocab)

    # for special in specials_list:
    #     tok.add(special)
    # tok.add(" [STOP]")
    # tok.save("SAM/model.pickle")
    # print(tok.vocab)
    # train()
    # train_sam()
    test_sam()

    


# " [WEATHER] [TIME] [TURRET] [MUSIC] [DATE] hey sam shoot fire intruder weather time what how red alert spotify music play song time what's how's"


# " <|begin_prompt|> <end_prompt|>"