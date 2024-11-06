import numpy as np

def derivative(img:np.ndarray, order):
    if order == 0:
        return img
    else:
        img_1 = derivative(img, order-1)
        Sn = np.zeros((img_1.shape[0], img_1.shape[1]-1))
        Sn = img_1[:, 1:] - img_1[:, :-1]
        return Sn
