import regex as re
import pickle
import sys
sys.path.append("../Nate")
from helpers import get_stats, merge
from model import Config
import torch
#regex.compile(pattern)
# https://github.com/Zjh-819/LLMDataHub



class Tokenizer:
    """
    Regex based bpe tokenizer
    The regex pattern is used to split the input text in encode()

    """

    def __init__(self):
        self.pattern = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\sp{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        
        self.special_tokens = []
        self.num_base_tokens = Config.vocab_size - len(self.special_tokens)
        self.special_tokens_dict = {
            self.num_base_tokens+i:token for i,token in enumerate(self.special_tokens)
        }
        self.special_tokens_inv = {
            token:self.num_base_tokens+i for i,token in enumerate(self.special_tokens)
        }
        self.merges = {}
        self.vocab = {}
        self.vocab_size = 256
        
        

    def train(self, text, vocab_size, loaded=False):
        print('training')
        """
        Since we are encoding our tokens to utf-8, and there are 256 base "tokens" in utf-8, if our vocab size is less
        than that, something is wrong or we have no need for a tokinzer. To get a vocab size greater than 256, we'll
        need to do merges of these utf-8 tokens and the number of merges we'll have to do to get our desired vocab size
        is vocab_size - 256.
        pieces are the regex-split pieces of our text that will be what we are encoding. This pattern
        (credits to Meta and the llama3 team) will make sure that we don't create tokens across text and 
        punctuation, or numbers, or any other non-letter symbols. 
        We encode our pieces to utf-8 and called these encodes pieces ids
        vocab holds a dictionary of form (int, bytes) where the bytes of each int are equivalent to the utf-8
        encoding of of each character at int. For example: vocab has 65: b'a' and the utf-8 encoding of a is 65, so when
        we do vocab[65], we get the byte value of a
        """
        assert vocab_size >= self.vocab_size
        num_merges = vocab_size - self.vocab_size
        

        pieces = re.findall(self.pattern, text)

        ids = [list(chars.encode('utf-8')) for chars in pieces]

        
        vocab = {idx:bytes([idx]) for idx in range(256)}

        self.vocab = vocab
        print(len(ids))
        for i in range(num_merges):
            stats= {}
            for piece in ids:
                get_stats(piece, stats)
            
            # if(len(stats) <= 5):
            #     print(f"{i}: Stats: {stats} Ids length: {len(ids)}")
            pair = max(stats, key=stats.get)
            
            idx = self.vocab_size + i
            ids = [merge(piece, pair, idx) for piece in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        
        self.vocab_size = vocab_size
        print(len(ids))
        

    def save(self, file=None):
        model = {}
        model['vocab'] = self.vocab
        model['merges'] = self.merges
        file = "model.pickle" if file is None else file
        stream = open(file, 'wb')
        pickle.dump(model, stream)
        stream.close()

    
    def load(self, file=None):
        
        file = "model.pickle" if file is None else file
        stream = open(file, 'rb')
        model = pickle.load(stream)
        self.vocab = model['vocab']
        self.merges = model['merges']
        stream.close()
        self.vocab_size = len(self.vocab)


    def add(self, token:str):
        #see what we can encode in the new token
        ids = self.encode_ordinary(token)
        #get stats for token
        
        while(len(ids) > 1):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            #with new pair, add one to vocab size, add merge to self.merges and self.vocab
            ids = merge(ids, pair, self.vocab_size)
            self.merges[pair] = self.vocab_size
            self.vocab[self.vocab_size] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_size += 1
            #go until ids is a single token (length <= 1)
            

    def encode_many(self, texts:tuple):
        out = []
        for text in texts:
            tokens = self.encode_ordinary(text)
            out.append(tokens)
        return out
        
                

    def encode_bytes(self, text_bytes):
        ids = list(text_bytes)
        """
        My thought process:
        We have the basic tokens for each character, and we want to map these to our vocab. To do this, we need to 
        find which sets of base tokens map to which merged tokens. Since this is in a loop, we can first get the first
        merge token the pair goes to, then we'll then iterate over again and will check if this merged token is a part of
        another merge token, and so on. So first in the iteration, let's just find how common each token pair is so that
        we can get the token pair for the lowest index merge token (index 256 in this case). We go until we find a 
        pair that is not in merges, meaning we have gone through all of our merged tokens and have fully encoded it.
        If the pair is in merges, then we have not found all of instances of token patterns that form our vocabulary,
        and we will merge ids with the index of the merge token that corresponds to the found token pair.
        """
        while len(ids) >= 2: #if ids is 1 or two, then we don't need to encode it any further
            stats = get_stats(ids) #Find how frequent each token pair is
            pair = min(stats, key= lambda pair: self.merges.get(pair,float('inf')))#find the token pair with the lowest index in self.merges that is in the text to be encoded
            if pair not in self.merges:
                break
            
            idx = self.merges[pair] 
            ids = merge(ids, pair, idx)
        return ids




    def encode_ordinary(self, text):
        """
        Encode the text to utf-8 tokens and then pass to encode_bytes to get encoded sequence
        """
        pieces = re.findall(self.pattern, text)
        ids = []
        for piece in pieces:
            base_tokens = piece.encode('utf-8')
            encoded = self.encode_bytes(base_tokens)
            ids.extend(encoded)

        return ids


    def encode_special(self, text, allowed_special="none_strict"):
        """
        allowed_special can be none | none_strict | all. If it's none, then special tokens
        in the text will be treated like normal text. If none_strict, if a special token is 
        in the text, an error will be raised. If all, we will process special tokens
        as special tokens.
        """
        """
        Thought Process: Use regex to split the text at each special token. Encode each piece and 
        insert these encoded pieces into final ids
        """
        
        if allowed_special == "none":
            return self.encode_ordinary(text)
        
        pattern = "("+"|".join(re.escape(token) for token in self.special_tokens)+")"
        
        text = re.split(pattern, text)
        
        if allowed_special == "none_strict" and len(text) > 1:
            raise ValueError("Text contains special tokens which are not allowed in prompt. \n Any of the following tokens are not allowed in text:"+'\n'.join(self.special_tokens))

        ids = []
        for piece in text:
            if piece in self.special_tokens:
                ids.append[self.special_tokens_dict[piece]]
            else:
                ids.extend(self.encode_ordinary(piece))
        
        return ids
        

    def decode(self, ids):
        """
        Thought Process:
        We have a list of tokens and these tokens correspond to some string value. These tokens are mapped to this 
        string value in self.vocab. We just need to go over each token, get the string value, and append as we go.
        """
        pieces = []

        for idx in ids:
            if idx in self.vocab:
                pieces.append(self.vocab[idx])
            elif idx in self.special_tokens:
                pieces.append(self.special_tokens_inv[idx])
            else:
                raise ValueError(f"Invalid token: {idx}")
        text = b"".join(pieces)
        text = text.decode('utf-8', errors='replace')
        return text

#     def test(self, string):
#         """
#         full_pattern: first part (?i:'s|'t|'re|'ve|'m|'ll|'d) gets all of the apostrophe-character pairs. Next part |[^\r\n\p{L}\p{N}] gets all of the white spaces and puctation. ?p{L}+ gets all character following these whitespaces or punctuations until the next 
#         """
#         full_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

#         test_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\sp{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
#         comp = re.compile(test_pattern)
#         res = comp.findall(string)
#         print(f"Characters: {len(string)} Tokens: {len(res)}")
#         print(res)

# if __name__ == "__main__":
#     tok = Tokenizer()
#     tok.test("Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other 'longest texts ever' on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing.")
