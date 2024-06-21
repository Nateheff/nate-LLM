from model import Nate, Config
from Tokenizer import Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate():
    m = Nate(Config)
    
    checkpoint = torch.load("model.pt")

    m.load_state_dict(checkpoint["model_state_dict"])
    model = m.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    tok = Tokenizer()
    tok.load()

    prompt = "Hello, my name is"
    tokens = tok.encode_ordinary(prompt)
    tokens = torch.tensor(tokens)
    tokens = tokens.view(1,tokens.shape[0])
    tokens = tokens.to(device)

    outputs = model.generate(tokens, 512)
    final = outputs[0].tolist()
    text = tok.decode(final)
    print(text)

generate()