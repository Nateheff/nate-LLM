import torch

def get_stats(ids: str, counts=None):
    """
    Returns the frequency of each token pair found in the tokinizer training data.
    """
    # print(ids)
    counts = {} if counts is None else counts
    for idx in zip(ids, ids[1:]):
        counts[idx] = counts.get(idx, 0) + 1
    
    return counts

def merge(ids, pair:tuple, idx:int):
    """
    Iterate through current ids and replace each instance of (pair) in ids with idx.
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids
    

def get_batch(tokens, batch_size: int, block_size: int, device):
    """
    Token set (either train or test) are passed in and we randomly select batch_size indexes in our
    dataset and we take block_size (context_length) characters as our x and the next values (index + 1) as our ys.
    """
    
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i + 1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y


def create_targets(x: tuple, y:tuple):
    #concatenate each prompt with the final target
    x = [prompt + target for prompt, target in zip(x,y)]

    return x

def pad(x:list, y:list, context_length:int, pad_token: int, padding_side="left"):
    for prompt, response in zip(x,y):
        padding = [pad_token]*(context_length - len(prompt))
        if padding_side == "right":
            prompt.extend(padding)
            response.extend(padding)
        else:
            prompt[:] = padding + prompt
            response[:] = padding + response

    

@torch.no_grad
def estimate_loss(model):
    pass




#get target full
#tokenize
#remove first token from targets
#pad prompts and targets
#done