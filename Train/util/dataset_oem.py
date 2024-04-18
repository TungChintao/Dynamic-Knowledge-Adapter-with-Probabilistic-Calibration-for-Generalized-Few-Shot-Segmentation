import os
import os.path

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# -------------------------- Pre-Training --------------------------

class BaseData(Dataset):
    def __init__(self, mode=None, data_root=None, data_list=None, data_set=None, transform=None, main_process=False, batch_size=None):

        assert data_set in ['oem']
        assert mode in ['train', 'val']

        self.num_classes = 7

        self.mode = mode
        self.data_root = data_root
        self.batch_size = batch_size

        self.class_list = list(range(1, 8))
        self.sub_list = self.sub_val_list = self.class_list


        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        self.data_list = []
        list_read = open(data_list).readlines()
        print("Processing data...")

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            line = line.strip()
            label_name = os.path.join(self.data_root, line)
            image_name = label_name.replace('labels', 'images')
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError('Image & label shape mismatch: ' + image_path + ' ' + label_path + '\n'))
        #label_class = np.unique(label).tolist()
        #print(label_class)
        
        raw_label = label.copy()

        if self.transform is not None:
            image, label = self.transform(image, label)
            #print(image.shape, label.shape)
        #print(label)
        if self.mode == 'val' and self.batch_size == 1:
            return image, label, raw_label
        else:
            return image, label
