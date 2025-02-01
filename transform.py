import math
import random
import torch
import numpy as np
import scipy.interpolate as si


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandInterpolation(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, csi):
        if random.random() > self.p:
            return csi
        csi = csi.squeeze()
        size = csi.shape
        x = np.arange(0, size[1], 1)
        dat = csi
        rand_start = random.randint(0, 200)
        for i in range(0, size[0]):
            data = csi[i, :]
            f = si.interp1d(x, data)
            x_new = np.linspace(rand_start, rand_start+199, 400)
            data_new = f(x_new)
            dat[i, :] = torch.from_numpy(data_new)
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandSample(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, csi):
        if random.random() > self.p:
            return csi
        csi = csi.squeeze()
        size = csi.shape
        rand_freq = random.randint(2, 4)
        data = csi[:, 0:size[1]:rand_freq]
        dat = data
        while dat.shape[1] < size[1]:
            dat = torch.cat((dat, data), dim=1)
        dat = dat[:, 0:size[1]]
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandTimeWarping(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, csi):
        if random.random() > self.p:
            return csi
        csi = csi.squeeze()
        size = csi.shape
        dat = csi
        rand_start = random.randint(0, size[1]/2)
        rand_end = random.randint(size[1]/2, size[1]-1)
        num1 = random.randint(size[1]/4, size[1]/2)
        num2 = random.randint(size[1]/4, size[1]/2)
        num3 = size[1]-num1-num2
        x = np.arange(0, size[1], 1)
        for i in range(0, size[0]):
            data = dat[i, :]
            f = si.interp1d(x, data)
            beg = np.linspace(0, rand_start, num1)
            mid = np.linspace(rand_start, rand_end, num2)
            end = np.linspace(rand_end, size[1]-1, num3)
            data_begin = f(beg)
            data_mid = f(mid)
            data_end = f(end)
            dat_new = torch.from_numpy(data_mid)
            dat_beg = torch.from_numpy(data_begin)
            dat_end = torch.from_numpy(data_end)
            dat[i, :] = torch.cat((dat_beg, dat_new, dat_end), dim=0)
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandWarping(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, csi):
        if random.random() > self.p:
            return csi
        csi = csi.squeeze()
        size = csi.shape
        dat = csi
        d = random.random() * 2 * math.pi
        x = torch.linspace(0, 2 * math.pi, size[1])
        x = x + d
        y = []
        for i in x:
            y.append(math.cos(i))
        warp = torch.tensor(y)/4+0.75
        for i in range(0, size[0]):
            dat[i, :] = dat[i, :] * warp
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandOverturn(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, csi):
        if random.random() > self.p:
            return csi
        csi = csi.squeeze()
        size = csi.shape
        dat = csi
        for i in range(0, size[0]):
            mean = torch.mean(dat[i, :])
            dat[i, :] = (dat[i, :]-mean)*-1
            dat[i, :] = dat[i, :] + mean
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandNoise(torch.nn.Module):

    def __init__(self, p, scale):
        super().__init__()
        self.p = p
        self.scale = scale

    def forward(self, csi):
        if random.random() > self.p:
            return csi
        csi = csi.squeeze()
        size = csi.shape
        dat = csi
        noise = np.random.normal(0, self.scale, size[1])
        for i in range(0, size[0]):
            dat[i, :] = dat[i, :] + noise
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, csi):
        csi = csi.squeeze()
        size = csi.shape
        dat = csi
        data = torch.abs(dat)
        max_data = data.max()
        dat = dat/max_data
        dat = torch.reshape(dat, (1, size[0], size[1]))
        return dat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"