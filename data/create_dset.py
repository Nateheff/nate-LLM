from torch.utils.data import DataLoader
from data import Sam_Dataset, dataset_fire
import numpy as np

dset = Sam_Dataset(dataset_fire, noise_src="data_talk.txt", src_length=48430)


loader = DataLoader(dataset=dset, batch_size=2, shuffle=True)

num_epochs = 100

for epoch in range(num_epochs):
    for x,y in loader:
        print(f"{x}: {y}")