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

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_filenames():
    img_list = [] #sorted(glob.glob("out/image/*"))
    mask_list = [] #sorted(glob.glob("out/mask/*"))

    for i in range(1, 5):
        img_list += sorted(glob.glob("out/image/*_" + str(i) + ".png"))
        mask_list += sorted(glob.glob("out/mask/*_" + str(i) + ".png"))

    return img_list, mask_list

def load_split_sets():
    print("Loading same train-val-test split as Double U-Net..")
    with open("train_set.pkl", 'rb') as f:
        train_set = pkl.load(f)
    with open("val_set.pkl", 'rb') as f:
        val_set = pkl.load(f)
    with open("test_set.pkl", 'rb') as f:
        test_set = pkl.load(f)
    
    return train_set, val_set, test_set

def generate_batches(train_set, batch_size):
    #Divide Train Data Into List of Batches for Training Loop
    train_loader_x = []
    train_loader_y = []

    for idx in range(0, len(train_set), batch_size):
        if idx + batch_size > len(train_set):
            x_tup, y_tup = list(zip(*(list(train_set)[idx:])))
        else:
            x_tup, y_tup = list(zip(*(list(train_set)[idx:idx + batch_size])))
        train_loader_x.append(x_tup)
        train_loader_y.append(y_tup)
    
    return train_loader_x, train_loader_y


def train(train_loader_x, train_loader_y, batch_size, num_batches, num_epochs, learning_rate):
    msrf_net = MSRF(in_ch = 3, n_classes = 2, init_feat = 32).to(device)
    optimizer = optim.Adam(msrf_net.parameters(), lr = learning_rate, betas=(0.9, 0.999), eps=1e-8)
    criterion  = CombinedLoss()

    for epochs in tqdm(range(num_epochs)):
        running_loss = 0
        for idx in tqdm(range(num_batches)):
            # Array of CV2 images
            cv2_img_data = [euler_utils.read_img(ele) for ele in train_loader_x[idx]]

            img_data = [euler_utils.img_to_tensor(cv2_img) for cv2_img in cv2_img_data]
            img_data = torch.cat(img_data, dim = 0).to(device)
            
            canny_data = [torch.FloatTensor(np.asarray(cv2.Canny(cv2.imread(ele), 10, 100), np.float32)/255.0) for ele in train_loader_x[idx]]
            canny_data = [t.view(-1, t.shape[0], t.shape[1]) for t in canny_data]
            canny_data = torch.stack(canny_data, dim = 0).to(device)

            mask_data = [euler_utils.mask_to_tensor(euler_utils.read_mask(ele)) for ele in train_loader_y[idx]]
            mask_data = torch.cat(mask_data, dim = 0).to(device)
            
            pred_1, pred_2, pred_3, pred_canny = msrf_net.forward(img_data.float(), canny_data)

            del img_data
            loss = criterion(pred_1, pred_2, pred_3, pred_canny, mask_data, canny_data)
            del pred_1, pred_2, pred_3, pred_canny, mask_data, canny_data
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            running_loss += float(loss.detach())
            del loss

            print('*************')

        print(f"For epoch {epochs + 1} MSRF loss is {running_loss}")
        if (epochs + 1) % 5 == 0:
            torch.save(msrf_net.state_dict(), f'msrf_cvc-clinic_{epochs+1}_{running_loss}.pt')

    # Save PyTorch model to disk
    model_file = 'msrf_cvc-clinic.pt'
    torch.save(msrf_net.state_dict(), model_file)
    print('Finished training! Model saved to ' + model_file)


def main():
    ########## Hyperparameters ##############
    learning_rate = 1e-4
    num_epochs = 120
    batch_size = 8
    #########################################

    # Change directory to /srv/home/kanbur/MSRF_Net
    os.chdir('/srv/home/kanbur/MSRF_Net')

    img_list, mask_list = generate_filenames()
    img_data = list(zip(img_list,mask_list))

    train_set, val_set, test_set = load_split_sets()
    num_batches = math.ceil(len(train_set)/batch_size)

    train_loader_x, train_loader_y = generate_batches(train_set, batch_size)

    train(train_loader_x, train_loader_y, batch_size, num_batches, num_epochs, learning_rate)

if __name__ == "__main__":
    main()