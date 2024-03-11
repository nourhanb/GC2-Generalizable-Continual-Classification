import os, sys
import warnings; warnings.filterwarnings("ignore")
import pytorch_lightning as pl
pl.seed_everything(23)

import argparse
import glob
import cv2, numpy as np
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import torchvision
import albumentations as A, albumentations.pytorch as AT
import tqdm
import itertools