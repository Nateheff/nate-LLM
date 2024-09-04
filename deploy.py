import torch
from model.model import Nate, Config
# from fastapi import FastAPI
from flask import Flask, request, jsonify
# from pydantic import BaseModel
from model.Tokenizer import Tokenizer

from vosk import Model, KaldiRecognizer
import json
import base64
import numpy as np



sam = None
speech2text = None

tok = None
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

def load_models():
    global sam, tok, speech2text

    base = Model("vosk-model-en-us-daanzu-20200905/vosk-model-en-us-daanzu-20200905")
    speech2text = KaldiRecognizer(base, 16000)
    speech2text.SetWords(True)



    sam = Nate(Config)
    optimizer = torch.optim.Adam(sam.parameters(), lr=1e-4)
    checkpoint = torch.load("SAM/model.pt")
    sam.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    sam.eval()
    sam = sam.to(device)

    tok = Tokenizer()
    tok.load("SAM/model.pickle")





def make_prompt(prompt:str):
    return "<|begin|>" + prompt + " <|end|>"



@app.route("/")
def home_endpoint():
    return "Hello World!"



@app.errorhandler(415)
def unsupported_media_type(e):
    return jsonify(error=f"Unsupported Media Type: Type must be application/json.\n{str(e)}"), 415


@app.route("/speech", methods=["POST"])
def speech():
    data = request.get_json()
    frames = data["speech"]
    frames = [base64.b64decode(frame) for frame in frames]
    speech2text.AcceptWaveform(b''.join(frames))
    results = speech2text.Result()
    text = json.loads(results)["text"]
    return text



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

        logits: torch.Tensor = sam.generate(toks,1)
        logits.to("cpu")
        print(tok.decode(logits.tolist()[0]))
        print(logits)
        return str(logits.tolist()[0][-1])




if __name__ == "__main__":
    from waitress import serve

    load_models()
    serve(app=app, host="0.0.0.0", port=8000)

        