import os
import os.path

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import Compose, Resize, ToTensor, Normalize


# -------------------------- Pre-Training --------------------------

class BaseData(Dataset):
    def __init__(self, args):

        self.data_root = args.rl_root

        self.data_list = []
        list_read = open(args.rl_list).readlines()
        print("Processing re-labeling data...")
    
        for l_idx in tqdm(range(len(list_read[:args.relabel_num]))):
            line = list_read[l_idx]
            line = line.strip()
            image_name = os.path.join(self.data_root, line)
            label_name = image_name.replace('images', 'labels')
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = Compose([Resize(args.image_size),
                                  ToTensor(),
                                  Normalize(mean=args.mean, std=args.std)])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            image = self.transform(image, None)

        return image
