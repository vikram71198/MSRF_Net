from euler_msrf import MSRF as MSRF
from euler_loss import CombinedLoss as CombinedLoss
import euler_utils
import pickle as pkl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import cv2
import glob
from tqdm import tqdm as tqdm
import pickle as pkl
import os
from matplotlib import pyplot as plt
import shutil

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change directory to /srv/home/kanbur/MSRF_Net
os.chdir('/srv/home/kanbur/MSRF_Net')

msrf_net = MSRF(in_ch = 3, n_classes = 2, init_feat = 32)
msrf_net.load_state_dict(torch.load("trained_models/msrf_cvc-clinic.pt"))#, map_location=torch.device('cpu')))
msrf_net = msrf_net.to(device)


def load_split_sets():
    print("Loading same train-val-test split as Double U-Net..")
    with open("train_set.pkl", 'rb') as f:
        train_set = pkl.load(f)
    with open("val_set.pkl", 'rb') as f:
        val_set = pkl.load(f)
    with open("test_set.pkl", 'rb') as f:
        test_set = pkl.load(f)
    
    return train_set, val_set, test_set

def infer(input_filenames):
    print('Files = ' + str(input_filenames))
    x = [euler_utils.img_to_tensor(euler_utils.read_img(ele[0])) for ele in input_filenames]
    x = torch.cat(x, dim = 0).to(device)
    x = x.float()
    #print(f'X Shape = {x.shape}')

    canny_x = [torch.FloatTensor(np.asarray(cv2.Canny(cv2.imread(ele[0]), 10, 100), np.float32)/255.0) for ele in input_filenames]
    #cv2.imwrite('canny.png', canny_x[0].numpy())
    canny_x = [t.view(-1, t.shape[0], t.shape[1]) for t in canny_x]
    canny_x = torch.stack(canny_x, dim = 0).to(device)
    canny_x = canny_x * 0

    with torch.no_grad():
        y_1, y_2, y_3, y_canny = msrf_net(x, canny_x)
        y_3 = y_3.detach().cpu()
        y_3 = F.softmax(y_3, dim=1)[0]
        #y_3 = (y_3 <= 0.5).int()
        y_3 = torch.argmax(y_3, dim=0)
    #print(f'Y3 shape = {y_3.shape}')
    #print(y_3)
    mask_inference = y_3.detach().cpu().numpy()*(255)
    cv2.imwrite('infer.png', mask_inference)
    shutil.copyfile(input_filenames[0][1], 'infer_exp.png')


train_set, val_set, test_set = load_split_sets()
infer([test_set[2]])