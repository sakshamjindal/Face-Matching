import random
import numpy as np
import torch

from backbone.dataset import Face_Dataset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn


from backbone.loss import ContrastiveLoss
from backbone.networks.inception_resnet_v1 import InceptionResnetV1
from backbone.trainer import train_epoch,validate_epoch,save_model

batch_size = 32
lr = 1e-2
n_epochs = 20
log_interval = 50

face_data_train = Face_Dataset(root_dir= "lfw/", file_root = "files/", train=True)
face_data_val =   Face_Dataset(root_dir= "lfw/", file_root = "files/", test=True)

Siamese_network = InceptionResnetV1(pretrained='vggface2')
checkpoint = torch.load("../pretrained/20180402-114759-vggface2.pt")
Siamese_network.load_state_dict(checkpoint)
Siamese_network = Siamese_network.cuda()

criterion = ContrastiveLoss()
optimizer = optim.Adam(Siamese_network.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 4, gamma=0.1, last_epoch=-1)


train_dataloader = DataLoader(face_data_train,
                        shuffle=True,
                        num_workers=8,
                        batch_size=batch_size)

test_dataloader = DataLoader(face_data_val,
                        shuffle=True,
                        num_workers=8,
                        batch_size=batch_size)

train_losses = []
val_losses = []
best_loss = np.inf

exp_name = "inception_resnet_V1_pretrained"

for epoch in range(20):
        
    learning_rate = scheduler.get_lr()
    train_loss = train_epoch(train_dataloader,Siamese_network,criterion,optimizer)
    val_loss = validate_epoch(test_dataloader,Siamese_network,criterion)
    scheduler.step()
    
    if val_loss<best_loss:
        best_loss =  val_loss
        save_model(Siamese_network,exp_name)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print("Epoch Number : {}".format(epoch))
    print("---------------------------------------------------------------")
    print("    Train Loss :{}  , Val_Loss : {} , Learning Rate: {}".format(train_loss,val_loss,learning_rate))
    print("\n")


