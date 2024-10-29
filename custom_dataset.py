import torch
import os
import json 

from torch.utils.data import Dataset

def _get_dataset(dataset_dir):
    with open(os.path.join(dataset_dir, 'anno.json'), 'r') as f:
        label_dict = json.load(f)

    sample_ls = []

    # ADHD
    adhd_dir = os.path.join(dataset_dir, 'ADHD')
    for image_name in os.listdir(adhd_dir):
        image_path = os.path.join(adhd_dir, image_name)

        elem = (image_path, label_dict[image_path])
        sample_ls.append(elem)

    # ASD 
    asd_dir = os.path.join(dataset_dir, 'ASD')
    for image_name in os.listdir(asd_dir):
        image_path = os.path.join(asd_dir, image_name)

        elem = (image_path, label_dict[image_path])
        sample_ls.append(elem)

    # Normal 
    normal_dir = os.path.join(dataset_dir, 'Normal')
    for image_name in os.listdir(normal_dir):
        image_path = os.path.join(normal_dir, image_name)

        elem = (image_path, label_dict[image_path])
        sample_ls.append(elem)

    return sample_ls

class FundusNeuroDevDataset(Dataset):
    def __init__(self, dataset_dir, transform):
        super(FundusNeuroDevDataset, self).__init__()
        
        self.samples = _get_dataset(dataset_dir)
        self.transform = transform

    def __getitem__(self, idx):
        x, label = self.samples[idx]

        x_i, x_j = self.transform(x)
        y = torch.LongTensor([label]).squeeze()

        return x_i, x_j, y
