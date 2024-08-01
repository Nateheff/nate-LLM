import torch
from model.model import Nate, Config
# from fastapi import FastAPI
from flask import Flask, request, jsonify
# from pydantic import BaseModel
from model.Tokenizer import Tokenizer


model = None
tok = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# app = FastAPI()
app = Flask(__name__)

def load_model():
    global model, tok

    model = Nate(Config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = torch.load("SAM/model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    model = model.to(device)

    tok = Tokenizer()
    tok.load("SAM/model.pickle")

# load_model()

def make_prompt(prompt:str):
    return "<|begin|>" + prompt + " <|end|>"

# class Post(BaseModel):
#     text: str


# @app.get("/")
@app.route("/")
def home_endpoint():
    return "Hello World!"


@app.errorhandler(415)
def unsupported_media_type(e):
    return jsonify(error=f"Unsupported Media Type: Type must be application/json.\n{str(e)}"), 415


@app.route("/test")
def test():
    return f"{request.method} TEST"

# @app.get("/predict")
# def predict(post: Post):

#     prompt = make_prompt(post.text)
#     toks = tok.encode_ordinary(prompt)
#     toks = torch.tensor(toks)
#     toks = toks.unsqueeze(0)
#     toks = toks.to(device)

#     logits: torch.Tensor = model.generate(toks,1)
#     logits.to("cpu")
#     print(tok.decode(logits.tolist()[0]))
#     print(logits)
#     return str(logits.tolist()[0][-1])


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = request.get_json()
        query = data["text"]
        prompt = make_prompt(query)
        toks = tok.encode_ordinary(prompt)
        toks = torch.tensor(toks)
        toks = toks.unsqueeze(0)
        toks = toks.to(device)

        logits: torch.Tensor = model.generate(toks,1)
        logits.to("cpu")
        print(tok.decode(logits.tolist()[0]))
        print(logits)
        return str(logits.tolist()[0][-1])


if __name__ == "__main__":
    from waitress import serve
    load_model()
    serve(app=app, host='0.0.0.0', port=5000)