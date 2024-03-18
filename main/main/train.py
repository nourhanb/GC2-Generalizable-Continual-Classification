import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models import *
from engines import train_fn

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/H-D/{}/*/".format("All"), 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/H-D/{}/*/".format("MSK_val"), 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = False, 
    ), 
}
FT = torch.load(
    "../../warmup/ckps/H-D/{}/best.ptl".format("MSK_val"), 
    map_location = "cpu", 
)
for parameter in FT.parameters():
    parameter.requires_grad = False
models = {
    "FT":FT, "FS":fcn_resnet50(), 
    "GS":fcn_3x64_gctx(), 
}

save_ckps_dir = "../ckps/H-D/{}".format("MSK_val")
if not os.path.exists(save_ckps_dir):
    os.makedirs(save_ckps_dir)
train_fn(
    train_loaders, num_epochs = 25, 
    models = models, 
    device = torch.device("cuda"), 
    save_ckps_dir = save_ckps_dir, 
)