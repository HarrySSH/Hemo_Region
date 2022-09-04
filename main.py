from __future__ import print_function

import argparse
# importing the libraries
import os
import sys
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import cv2
import  glob
import time
import albumentations
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
### Internal Imports
from models.models import Myresnext50, Myresnext50_knowledge
from train.train_classification import trainer_classification
from utils.utils import configure_optimizers
from Datasets.DataLoader import Img_DataLoader

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob

def main(args):

    img_dir = args.train_dir
    # I only use the normal images
    image_files = glob.glob(img_dir + '*/*.png')
    labels = [x.split('/')[-2] for x in image_files]


    X_train = glob.glob(os.path.join(args.train_dir,'*/*'))

    X_val = glob.glob(os.path.join(args.val_dir,'*/*'))


    labels = [x.split('/')[-2] for x in X_train]
    types = set(labels)

    types = list(cell_types)
    types.sort()



    types_df = pd.DataFrame(types, columns=['Types'])# converting type of columns to 'category'
    types_df['Types'] = types_df['Types'].astype('category')# Assigning numerical values and storing in another column
    types_df['Cell_Types_Cat'] = types_df['Types'].cat.codes



    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
    enc_df = pd.DataFrame(enc.fit_transform(types_df[['Types_Cat']]).toarray())# merge with main df bridge_df on key values
    types_df = types_df.join(enc_df)


    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # Interesting! This worked for no reason haha
    if args.input_model == 'ResNeXt50':
        resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=args.pretrained)
        my_extended_model = Myresnext50(my_pretrained_model= resnext50_pretrained, num_classes = len(cell_types))
    #### train_iteration_1
    #### Use can use this for now. but remember the best normalization number should be calculated from your own data
    transform_pipeline = albumentations.Compose(
        [
            albumentations.Normalize(mean=(0.5637, 0.5637, 0.5637), std=(0.2381, 0.2381, 0.2381)),

        ]
    )



    trainer = trainer_classification(train_image_files=X_train, validation_image_files=X_val, model=my_extended_model,
                                     img_transform=transform_pipeline, init_lr=args.init_lr,
                                     lr_decay_every_x_epochs=args.lr_decay_every_x_epochs,
                                     weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs, gamma=args.gamma,
                                     df=cell_types_df, df_features = features_clean_onehot, graph_loss = args.graph_loss,
                                     save_checkpoints_dir=args.save_checkpoints_dir)

    My_model = trainer.train(my_extended_model)


# Training settings
parser = argparse.ArgumentParser(description='Configurations for Model training')


parser.add_argument('--train_dir', type=str,
                    default='',
                    help='train data directory')

parser.add_argument('--val_dir', type=str,
                    default='',
                    help='val data directory')

parser.add_argument('--input_model', type=str,
                    default='ResNeXt50',
                    help='input model, the defulat is the pretrained model')



parser.add_argument('--pretrained', type=bool,
                    default=True,
                    help='the defulat is the pretrained model')

parser.add_argument('--init_lr', type=float,
                    default=0.001,
                    help='learning rate')

parser.add_argument('--weight_decay', type=float,
                    default=0.0005,
                    help='weight decay')

parser.add_argument('--gamma', type=float,
                    default=0.1,
                    help='gamma')

parser.add_argument('--epochs', type=float,
                    default=30,
                    help='epoch number')

parser.add_argument('--batch_size', type=int,
                    default=16,
                    help='epoch number')

parser.add_argument('--lr_decay_every_x_epochs', type = int,
                    default=10,
                    help='learning rate decay per X step')

parser.add_argument('--save_checkpoints_dir', type = str,
                    default=None,
                    help='save dir')





args = parser.parse_args()

if __name__ == "__main__":
    main(args)
    print('Done')