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
import random

random.seed(0)
torch.manual_seed(0)
#torch.use_deterministic_algorithms(True)

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change directory to /srv/home/kanbur/MSRF_Net
os.chdir('/srv/home/kanbur/MSRF_Net')

msrf_net = MSRF(in_ch = 3, n_classes = 2, init_feat = 32).eval().to(device)
msrf_net.load_state_dict(torch.load("trained_models/msrf_cvc-clinic.pt",  map_location=torch.device('cpu')))


def generate_splits():
    # Get all base images (no augmentations)
    img_list = sorted(glob.glob("out/image/*_1.png"))
    mask_list = sorted(glob.glob("out/mask/*_1.png"))

    img_data = list(zip(img_list,mask_list))
    data_len = len(img_data)
    train_set, val_set, test_set = torch.utils.data.random_split(img_data, [round(0.4*data_len), round(0.1*data_len), data_len - round(0.4*data_len) - round(0.1*data_len)])

    final_train_set = []
    final_val_set = []
    final_test_set = []
    for img, mask in train_set:
        img_prefix = img[:-5]
        mask_prefix = mask[:-5]
        final_train_set.append((img, mask))
        for i in range(2, 9):
            final_train_set.append((img_prefix + str(i) + ".png", mask_prefix + str(i) + ".png"))
    
    for img, mask in val_set:
        img_prefix = img[:-5]
        mask_prefix = mask[:-5]
        final_val_set.append((img, mask))
        for i in range(2, 9):
            final_val_set.append((img_prefix + str(i) + ".png", mask_prefix + str(i) + ".png"))
    
    for img, mask in test_set:
        img_prefix = img[:-5]
        mask_prefix = mask[:-5]
        final_test_set.append((img, mask))
        for i in range(2, 9):
            final_test_set.append((img_prefix + str(i) + ".png", mask_prefix + str(i) + ".png"))

    return final_train_set, final_val_set, final_test_set

def infer(input_filenames):
    print('Files = ' + str(input_filenames))
    x = [euler_utils.img_to_tensor(euler_utils.read_img(ele[0])) for ele in input_filenames]
    x = torch.cat(x, dim = 0)
    x = x.float()
    x = x.to(device)
    with open("x.pkl", 'wb') as f:
        pkl.dump(x, f)
    #print(f'X Shape = {x.shape}')

    canny_x = [torch.FloatTensor(np.asarray(cv2.Canny(cv2.imread(ele[0]), 10, 100), np.float32)/255.0) for ele in input_filenames]
    cv2.imwrite('canny.png', canny_x[0].numpy() * 255)
    canny_x = [t.view(-1, t.shape[0], t.shape[1]) for t in canny_x]
    canny_x = torch.stack(canny_x, dim = 0).to(device)
    with open("canny.pkl", 'wb') as f:
        pkl.dump(canny_x, f)
    #canny_x = canny_x * 0

    with torch.no_grad():
        y_1, y_2, y_3, y_canny = msrf_net(x, canny_x)
        with open("y_3.pkl", 'wb') as f:
            pkl.dump(y_3, f)
        y_3 = y_3.detach().cpu()
        y_3 = F.softmax(y_3, dim=1)[0]
        #y_3 = (y_3 <= 0.5).int()
        y_3 = torch.argmax(y_3, dim=0)
    #print(f'Y3 shape = {y_3.shape}')
    #print(y_3)
    mask_inference = y_3.detach().cpu().numpy()*(255)
    cv2.imwrite('infer.png', mask_inference)
    shutil.copyfile(input_filenames[0][1], 'infer_exp.png')


train_set, val_set, test_set = generate_splits()
infer([test_set[0]])#, test_set[1], test_set[2], test_set[3], test_set[4], test_set[5], test_set[6], test_set[7]])

with open("train_set.pkl", 'wb') as f:
    pkl.dump(train_set, f)

with open("test_set.pkl", 'wb') as f:
    pkl.dump(test_set, f)

print(len(train_set))

print(len(test_set))