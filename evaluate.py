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
import euler_metrics
from tqdm import tqdm as tqdm
from ignite.metrics import Precision, Recall

random.seed(0)

torch.manual_seed(0)

def infer(input_filenames, number):
    #print('Files = ' + str(input_filenames))
    x = [euler_utils.img_to_tensor(euler_utils.read_img(ele[0])) for ele in input_filenames]
    x = torch.cat(x, dim = 0)
    x = x.float()
    x = x.to(device)
    #print(f'X Shape = {x.shape}')
    # with open("x.pkl", 'rb') as f:
    #     x_og = pkl.load(f)

    # print(f"Are inputs equal ? {torch.equal(x, x_og)}")

    canny_x = [torch.FloatTensor(np.asarray(cv2.Canny(cv2.imread(ele[0]), 10, 100), np.float32)/255.0) for ele in input_filenames]
    #cv2.imwrite('canny.png', canny_x[0].numpy() * 255)
    canny_x = [t.view(-1, t.shape[0], t.shape[1]) for t in canny_x]
    canny_x = torch.stack(canny_x, dim = 0).to(device)
    # with open("canny.pkl", 'rb') as f:
    #     canny_og = pkl.load(f)

    # print(f"Is canny equal ? {torch.equal(canny_x, canny_og)}")

    with torch.no_grad():
        y_1, y_2, y_3, y_canny = msrf_net(x, canny_x)
        # with open("y_3.pkl", 'rb') as f:
        #     y_3_og = pkl.load(f)
        # print(f"Are predictions the same? {torch.equal(y_3_og, y_3)}")
        y_3 = y_3.detach().cpu()
        y_3 = F.softmax(y_3, dim=1)[0]
        y_3 = torch.argmax(y_3, dim=0)

    y_truth = torch.from_numpy(euler_utils.read_mask(input_filenames[0][1]))

    mask_inference = y_3.detach().cpu().numpy()*(255)
    cv2.imwrite(f'good_results_msrf/infer{number}.png', mask_inference)
    shutil.copyfile(input_filenames[0][1], f'good_results_msrf/infer_exp{number}.png')

    dice_loss = euler_metrics.dice_loss(y_truth, y_3)

    y_truth = (y_truth >= 0.5).int()

    y_3 = (y_3 >= 0.5).int()

    iou = euler_metrics.iou(y_truth, y_3)

    precision = Precision()

    precision.update((y_3, y_truth))

    prec = precision.compute()

    recall = Recall()

    recall.update((y_3, y_truth))

    rec = recall.compute()

    print(f'Dice Loss = {dice_loss}')
    print(f'IOU = {iou}')
    print(f'Precision = {prec}')
    print(f'Recall = {rec}')
    
    return dice_loss, iou, prec, rec


# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change directory to /srv/home/kanbur/double-u-net
os.chdir('/srv/home/kanbur/MSRF_Net')

msrf_net = MSRF(in_ch = 3, n_classes = 2, init_feat = 32).eval().to(device)
msrf_net.load_state_dict(torch.load("trained_models/msrf_cvc-clinic.pt",  map_location=torch.device('cpu')))

indices = [0, 1200, 1203, 1216, 1235, 1248, 1272, 1275, 1368]

train_set = []

test_set = []

with open("train_set.pkl", 'rb') as f:
    train_set = pkl.load(f)

with open("test_set.pkl", 'rb') as f:
    test_set = pkl.load(f)

# print(len(train_set))
# print(len(test_set))

avg_dice = 0
avg_iou = 0
avg_precision = 0
avg_recall = 0

for idx in tqdm(indices):
    if idx == 0:
        dice, iou, prec, rec = infer([test_set[idx]], idx)
    else:
        dice, iou, prec, rec = infer([train_set[idx]], idx)
    avg_dice += dice
    avg_iou += iou
    avg_precision += prec
    avg_recall += rec

print(f"Average Dice loss - {avg_dice/len(indices)}")
print(f"Average IOU - {avg_iou/len(indices)}")
print(f"Average Precision -{avg_precision/len(indices)}")
print(f"Average Recall - {avg_recall/len(indices)}")