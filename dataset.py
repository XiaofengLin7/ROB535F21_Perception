"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms
import ipdb
import time

class GTADataset(torch.utils.data.Dataset):
    def __init__(
            self, label_csv_file, bbox_csv_file, S=7, B=2, C=3, image_size=(448, 448)):
        self.label = pd.read_csv(label_csv_file)
        self.bbox = pd.read_csv(bbox_csv_file)
        self.S = S
        self.B = B
        self.C = C
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),])
        self.image_size = image_size
        self.images = torch.zeros((len(self.label), 3, *image_size))
        self.grid_widths = []
        self.grid_heights = []
        self.image_widths = []
        self.image_heights = []
        for i, (file_name, label) in enumerate(self.label.values):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Loading images:{i}/{len(self.label)}")
            image_name = './data/trainval/'+file_name + '_image.jpg'
            image = Image.open(image_name)
            self.grid_widths.append(image.size[0] / self.S)
            self.grid_heights.append(image.size[1] / self.S)
            self.image_widths.append(image.size[0])
            self.image_heights.append(image.size[1])
            if self.transform:
                image = self.transform(image)
            self.images[i] = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image = self.images[index]
        grid_width, grid_height = self.grid_widths[index], self.grid_heights[index]
        image_width, image_height = self.image_widths[index], self.image_heights[index]

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        label_matrix[:, :, 0] = torch.ones(label_matrix.shape[0:2], dtype=torch.float)
        name, label = self.label.iloc[index]
        if label != 0:
            x, y, w, h = self.bbox[self.bbox['guid/image'] == name].iloc[0][1:]
            x = x / image_width * self.image_size[0]
            w = w / image_width * self.image_size[0]
            y = y / image_height * self.image_size[1]
            h = h / image_height * self.image_size[1]
            idx_x = int(x//grid_width)
            idx_y = int(y//grid_height)
            label_matrix[idx_x][idx_y][0] = 0.0
            label_matrix[idx_x][idx_y][label] = 1.0
            label_matrix[idx_x][idx_y][self.C] = 1.0
            label_matrix[idx_x][idx_y][self.C+5] = 1.0
            norm_x =(x - idx_x * grid_width) / grid_width
            norm_y =(y - idx_y * grid_height) / grid_height
            norm_w = w / grid_width
            norm_h = h / grid_height
            label_matrix[idx_x][idx_y][self.C+1 : self.C+5] = torch.tensor([norm_x, norm_y, norm_w, norm_h])
            label_matrix[idx_x][idx_y][self.C+6 : self.C+10] = torch.tensor([norm_x, norm_y, norm_w, norm_h])

        return image, label_matrix

if __name__=='__main__':
    dataset = GTADataset('./data/trainval/trainval_labels.csv', 'data/trainval/trainval_bboxes.csv')

