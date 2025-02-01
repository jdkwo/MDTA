import os

import scipy.io as sio
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, data_folder, data_transform, test):
        self.test = test
        self.data_folder = data_folder
        self.data_name = os.listdir(self.data_folder)
        self.transform = data_transform
        self.position_map = ({
            '1': 0, '2': 1, '3': 4, '4': 2, '5': 3
        })
        self.orientation_map = ({
            '1': 0, '2': 1, '3': 2, '4': 3, '5': 4
        })
        self.user_map = ({
            '5': 0, '13': 1, '14': 2, '1': 0, '3': 1, '6': 2, '8': 3, '9': 4
        })

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, item):
        path = self.data_name[item]
        data_1 = sio.loadmat(os.path.join(self.data_folder, path))['csi']
        data = torch.tensor(data_1, dtype=torch.float)
        road = os.path.join(self.data_folder, path)
        split_road = road.split('/')
        csi_road = split_road[-1].split('-')
        num = int(csi_road[2])
        num = num - 1
        user = csi_road[1][4:]
        user = self.user_map[user]
        position = csi_road[3]
        position = self.position_map[position]
        orientation = csi_road[4]
        orientation = self.orientation_map[orientation]

        if self.test:
            if self.transform:
                image = self.transform(data)
                image = torch.reshape(image, (240, 400))
            else:
                image = data
                img = torch.reshape(image, (240, 400))
                image = img
                position = position - 4
                # orientation = orientation - 4
                #user = user - 4
                #position = position - 2
        else:
            if self.transform:
                image = self.transform(data)
                img0 = image[0]
                img1 = image[1]
                img2 = torch.reshape(img0, (240, 400))
                img3 = torch.reshape(img1, (240, 400))
                image = [img2, img3]
            else:
                image = data
                img = torch.reshape(image, (240, 400))
                image = img

        return image, num, position, orientation, user
