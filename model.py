import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Model Configuration Arguments

n_layers: The number of layers our transformer will have. A "layer" is composed of a multi-headed self-attention layer,
a feed-forward layer, a residual connection, and a normalization layer of the sum of the residaul connection.

n_heads: The number of attention heads per layer. Intuitively, multiple heads with different initializations will
learn different features of the input and the information/knowledge of all the heads are concatenated along the last dimension so that we end up with (batch_size, context_length, head_dim * n_heads) and we multiply this by an (n_embd, n_embd)
weight matrix, so we need out head_dim to be n_embd // n_heads to allow for this multiplication.

n_embd: The size of the embedding vectors for each token and position in our context. These are learnable parameters
that the model uses to gain an "understanding" of each token and position and it's how our model will know which token
is where.

head_dim: (My favorite parameter) The length of the vector of the key, query, and value for each token in 
our context. When we do our key * query to get our affinities matrix, I like to think of this dimension as how much
information is going into each value in the affinities matrix. For example, if this is 32, then each token in our 
context will have a 32-value vector representing it's key, query, and value. The larger this number, the longer this
vector and the more "information" can be held about each of these values.

vocab_size: This is as it sounds; it's how many tokens are in our vocabulary. Our model's final output will be of this
dimension since it will output the probability of the next token being each token in our vocabulary. 1k is pretty
tiny compared to modern models which have vocabularies of 100k+, but it's a tiny model, so this will suffice

ffn_dim_raise: This is the mutliplier by which we will raise the dimensions in our feed forward network.

norm_eps: Our epsilon value for our normalization layer. For those who aren't sure why this value exists, it 
avoids dividing by zero in out normalization by adding a very small value.

batch_size: You know this one :)

max_context_length: The number of tokens our model can "pay attention to" at one time. Literally, this is the size of 
affinities matrix.
"""

@dataclass
class ModelConfig:
    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 1024
    head_dim: int = n_embd // n_heads
    vocab_size: int = 1092
    ffn_dim_raise: int = 4
    norm_eps: int = 1e-8
    batch_size: int = 256
    max_context_length: int = 256
    dropout: float = 0.1

Config = ModelConfig()

class Attention(nn.Module):
    """
    Self-Attention

    Key, Query, Value: These is re matrice of dimenions (n_embd, head_size). The values in these matrices are learnable
    parameters that will be multipled by the input sequence of tokens to get the key, query, and value vector for each
    token in the input 

    Other parameters are explained above

    """

    def __init__(self, n_embd, head_size, context_length, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.context_length = context_length
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        First, we get the key, query, and value vector for each token

        Next, we get our intial affinities matrix (affintiies_full) by multiplying the keys and querys of each token.
        We mutliply these affinites by the scaling factor: 1/(sqrt(head_size))
        tril is a matrix of diagonal 1's and 0's that will be used the set the affinities of future tokens to -inf
        affinities.masked_fill(...), this is what makes this self-attention. We set all affinties in the upper right 
        "triangle" to -inf so that we do not have any affinity for future toekns, only contextual tokens.
        We then take the softmax by rows to get the "scaling factors" by which the values of each token should imapct
        each other token. 
        We multiply the values of each token by the affinities matrix to get the final output.
        The final output is the value of each token in the context of all the previous tokens.
        """
        length = x.shape[-2]
        assert length <= self.context_length
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        affinities_full = q @ k.transpose(-2,-1)
        affinities_full = affinities_full * self.head_size**-0.5
        tril = torch.tril(torch.ones((length, length), device=device))
        affinities = affinities_full.masked_fill(tril == 0, float('-inf')) #set the affinities at the locations where
        #trill == 0 to -inf so that we don't "pay attention" to future tokens. (-inf instead of 0 to make softmax
        #output better)
        affinities = F.softmax(affinities, dim=-1)
        out = self.dropout(affinities)
        out =  affinities @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Headed Attention

    heads: A Module_List of attention heads
    linear: A linear layer for the final output
    """
    def __init__(self, n_heads, n_embd, head_dim, context_length, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_size = head_dim
        self.conext_length = context_length
        self.heads = nn.ModuleList([Attention(n_embd,head_dim,context_length, dropout) for i in range(n_heads)])
        self.linear = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        We pass our input tokens through the heads and concatenate along the last dimension to get the "information"
        for each of the heads in output
        We then pass this full matrix of info from all of the heads through a linear layer 
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out


class FeedForward(nn.Module):
    """
    Feed Forward Layer

    Two fully connected layers with a ReLU non-linearity in the middle which are vital to the model.
    This is a perfect application of fully connected layers as they allow the model to learn complex 
    relationships and pattens better. This layer is also super important because of the non-linearity.
    This is the first non-linearity in the model. The model can now learn more complex, non-linear relationships
    in the data.
    """
    def __init__(self, n_embd, ffn_dim_raise, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*ffn_dim_raise),
            nn.ReLU(),
            nn.Linear(n_embd * ffn_dim_raise, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class RMSNorm(nn.Module):
    """
    RMS (Root Mean Square) Normalization
    https://arxiv.org/pdf/1910.07467

    Implementation based on paper linked above. 
    """

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor):
        # X / sqrt(mean(x^2) + eps)
        return x * (x.pow(2).mean(dim=-1, keepdim=True) + self.eps)**-0.5
    
    def forward(self, x):
        return self._norm(x) * self.weights
    

class Block(nn.Module):
    """
    Transformer Block

    Bring all of the sublayers together to one unfiied block
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config.n_heads, config.n_embd, config.head_dim, config.max_context_length, config.dropout)
        self.ffn = FeedForward(config.n_embd, config.ffn_dim_raise, config.dropout)
        self.norm_attn = RMSNorm(config.n_embd, config.norm_eps)
        self.norm_ffn = RMSNorm(config.n_embd, config.norm_eps)

    def forward(self, x):
        """
        Why are we adding x back? Residual Connecitons! Residual connections helps us not loose
        any information from the operations. We retain this information by adding x back to the 
        output.
        """
        x = x + self.self_attention(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x
        


class Nate(nn.Module):
    """
    Nate Model

    token_embedding: Each token in our vocabularly is assigned to a vector of length n_embd
    of learnable parameters that will be trained to be better representations of each token.

    position_embedding: Each position in max_context_length is assigned to a vector of length
    n_embd of learnable parameters that will be trained to hint to the model what position a token
    is at.
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding = nn.Embedding(self.config.max_context_length, self.config.n_embd)
        self.layers = nn.ModuleList([Block(config) for i in range(self.config.n_layers)])
        self.norm = RMSNorm(self.config.n_embd, self.config.norm_eps)
        self.linear = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(self, x: torch.tensor, targets=None):
        """
        We embed our input tokens then perform a forward pass. 
        Once we have our final outputs, we turn our batches of logit matrices and 
        target matrices into individual matrices we can use to calculate the loss.
        If there are no targets, this tells us we are performing inference, not training
        """
        B, T = x.shape
        toks = self.token_embedding(x)
        pos = self.position_embedding(torch.arange(T, device= device))


        x = toks + pos
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.linear(x)
        if targets is not None:
            logits = logits.view(logits.shape[0] * logits.shape[1], -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets.view(B*T))
        else:
            loss = 0

        return logits, loss

    def generate(self, prompt:torch.tensor, new_tokens: int) -> torch.tensor:

        for _ in range(new_tokens):
            
            idx_cond = prompt[:, -self.config.max_context_length:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            prompt = torch.cat((prompt, idx_next), dim=1) # (B, T+1)
        return prompt


