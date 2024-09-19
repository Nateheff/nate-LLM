from torch.utils.data import Dataset
import numpy as np

begin_token = "<|begin|>"
end_token = " <|end|>"

class Sam_Dataset(Dataset):
    def __init__(self, data:list, noise_src=None, src_length=None):
        self.data = data
        self.noise_src = noise_src
        self.src_length = src_length
        self.text = None
        if self.noise_src:
            f = open(self.noise_src, 'r')
            self.text = f.read()
            
        

    def __getitem__(self, index):
        if self.noise_src:
            ind_noise = np.random.randint(0,self.src_length-32)
            noise = self.text[ind_noise:ind_noise+32]
            beg = noise.find(' ')
            noise = noise[beg:]
            return begin_token + self.data['prompt'][index] + noise + end_token, self.data['response'][index]
        else:
            return begin_token + self.data["prompt"][index] + end_token, self.data["response"][index]
    
    def __len__(self):
        return len(self.data["prompt"])



dataset = {
    'prompt':[
        " hey sam what time is it",
        " sam what time is it",
        " hey sam got the time",
        " sam got the time",
        " sam what's the date",
        " hey sam what's the date",
        " sam what day is it",
        " hey sam what day is it",
        " hey sam play my spotify.",
        " hey sam play spotify",
        " sam play my spotify",
        " sam play spotify",
        " hey sam play music",
        " sam play music",
        " hey sam spotify",
        " sam spotify",
        " sam red alert",
        " hey sam red alert",
        " sam red alert",
        " hey sam red alert",
        " sam turret",
        " hey sam turret",
        " sam intruder",
        " hey sam intruder",
        " sam we got an intruder",
        " hey sam we got an intruder",
        " hey sam how's the weather",
        " sam how's the weather",
        " hey sam what's the weather like",
        " sam what's the weather like",
        " hey sam turret",
        " sam turret",
        " sam shoot",
        " hey sam shoot",
        " hey sam fire",
        " sam fire",
        " hey sam play a song",
        " Sam play a song",
        " sam music",
        " hey sam music",
        " sam is it hot outside",
        " hey sam is it hot outside",
        " hey sam what's it like outside",
        " sam what's it like outside",
        " sam what's the time",
        " hey sam what's the time",
        " sam time",
        " hey sam time",
        " sam stop",
        " sam quit",
        " sam off",
        " hey sam off",
        " hey sam stop",
        " hey sam quit",
    ],
    'response':[
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [DATE]",
        " [DATE]",
        " [DATE]",
        " [DATE]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [WEATHER]",
        " [WEATHER]",
        " [WEATHER]",
        " [WEATHER]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [MUSIC]",
        " [WEATHER]",
        " [WEATHER]",
        " [WEATHER]",
        " [WEATHER]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
    ]
}

dataset_fire = {
    'prompt':[
        " sam turret",
        " sam turret",
        " sam shoot",
        " sam shoot",
        " sam fire",
        " sam fire",
        " sam red alert",
        " sam red alert",
        " sam red alert",
        " hey sam red alert",
        " sam turret",
        " sam turret",
        " sam intruder",
        " sam intruder",
        " sam we got an intruder",
        " sam we got an intruder",
        " sam shoot her",
        " sam shoot him",
        " sam fuck him up",
        " sam fuck her up",
        " sam fuck em up",
        " sam kill",
        " sam kill him",
        " sam kill her",
        " sam kill them",
        " sam destroy",
        " sam destroy her",
        " sam destroy him",
        " sam shoot them", #29
        " sam track me",
        " sam track",
        " sam tracking",
        " sam watch me",
        " sam track her",
        " sam track him",
        " sam track them",
        " sam keep an eye on me",
        " sam heep an eye on them",
        " sam keep an eye on him",
        " sam watch her",
        " sam watch",
        " sam watch him",
        " sam watch them",
        " sam eyes up",
        " sam eyes open",
        " sam head on a swivel", #17
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam ",
        " sam " #20
    ],
    'response':[
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TURRET]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [TIME]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
        " [STOP]",
    ]
}

print(len(dataset['prompt']), len(dataset["response"]))

noise = [
    " watch",
    " stay",
    " check it out",
    " stay there",
    " say there",
    " watch this",
    " don't",
    " don't move",
    " stand",
    " stand there",
    " stand still",
    " close",
    " close your",
    " close your eyes",
    " run",
    " dude",
    " no",
    " no way",
    " sick"
]

dict_dataset = []

def create_dataset():
    prompts = dataset['prompt'].copy()
    responses = dataset['response'].copy()
    for prompt,response in zip(prompts, responses):
        
        for suffix in noise:
            dataset['prompt'].append(prompt+suffix)
            dataset['response'].append(response)
    
    dset = Sam_Dataset(dataset)
    return dset

def create_tok_dataset():
    create_dataset()
    prompts = ' '.join(dataset["prompt"])
    responses = ' '.join(dataset["response"])
    return prompts +" "+responses
    

def create_dict_dataset():
    create_dataset()
    for prompt, response in zip(dataset["prompt"], dataset["response"]):
        new_dict = {}
        new_dict['prompt'] = prompt
        new_dict['response'] = response
        dict_dataset.append(new_dict)
    dset = Sam_Dataset(dict_dataset)
    return dset


def create_dataset_fire():
    dset = Sam_Dataset(dataset_fire, noise_src="data_talk.txt", src_length=48430)
    return dset

if __name__ == "__main__":
    create_dataset()
    create_dict_dataset()
    print(dict_dataset)

    