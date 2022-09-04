# importing the libraries
import os
import sys
import pandas as pd
import torch
from torch.utils import data
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import  glob
import time
import albumentations 
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random

from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
sys.path.append("..")
from Datasets.DataLoader import Img_DataLoader

################################################
# Smooth Gradient
import saliency.core as saliency
from matplotlib import pylab as P
# Boilerplate methods.
def ShowImage(im):
    plt.axis('off')
    plt.imshow(im)

def ShowGrayscaleImage(im, title='', ax=None):

    plt.axis('off')
    plt.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)

def ShowHeatMap(im, ax=None):
    
    plt.axis('off')
    plt.imshow(im, cmap='inferno')

def LoadImage(file_path):
    #im = PIL.Image.open(file_path)
    im = cv2.imread(file_path,1)[:,:,::-1]
    #im = im.resize((96, 96))
    im = np.asarray(im)
    return im

def PreprocessImages(images= None):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    if len(images.shape) != 4:
        images = transform_pipeline(image=images)["image"]
        images = images.reshape((1,images.shape[0], images.shape[1], images.shape[2]))
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    
    return images.requires_grad_(True)

def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()
def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()

    
def call_model_function(images, call_model_args=None, expected_keys=None ):
    model = call_model_args['model']
    model = model.eval()
    images = PreprocessImages(images)
    target_class_idx =  call_model_args['class_idx_str']
    output = model(images)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs



########################################################

def Grad_CAM_plot(image_list= [], targets= None, model= None, 
            dataframe= None, transform= None):
    Orig_img = Img_DataLoader(img_list= image_list, split='viz',df= dataframe,transform = transform)
    shuffle = False
    dataloader = DataLoader(Orig_img, batch_size=32, num_workers=2, shuffle=shuffle)
    
    for image in dataloader:
        break
    # print prediction
    model.eval()
    res = model(image['image'])
    res = torch.flatten(res, start_dim=1).detach().cpu().numpy()
    
    #actual_name = image_list[0].split('/')[-2]
    actual_name = [x.split('/')[-2] for x in image_list]
    #predicted_name = dataframe[dataframe['Cell_Types_Cat']==np.where(index==1)[1][0]].iloc[:,0].tolist()[0]
    
    celltype_id = np.argmax(res, axis=1).tolist()
    
    predicted_name = [dataframe[dataframe['Cell_Types_Cat']==x].iloc[:,0].tolist()[0] for x in celltype_id]

    # print saliency map
    rows = 1
    columns = len(actual_name)
    
    fig = plt.figure(figsize=(columns*3, 3))
    
    input_tensor = image['image']
    # Construct the CAM object once, and then re-use it on many images:
    target_layers = [model.pretrained.layer4[-1]]
    targets = [ClassifierOutputTarget(5)]
    cam = GradCAM(model=model, 
                  target_layers=target_layers,  use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor,
                    aug_smooth=True,eigen_smooth=True)#, targets=targets)
    
    # In this example grayscale_cam has only one image in the batch:
    for _order in range(columns):
        fig.add_subplot(rows, columns, _order + 1)
        rgb_img = cv2.imread(image_list[_order], 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        
        grayscale_cam_one = grayscale_cam[_order, :,:]
        visualization = show_cam_on_image(rgb_img, grayscale_cam_one, use_rgb=True)
        
        plt.imshow(visualization)
        plt.axis('off')
        plt.title('Groundtruth:'+ actual_name[_order]+'\n' + 'Prediction:'+ ' '+predicted_name[_order])
        
        
        

def Saliency_gray_plot(model = None, image_list =None, df = None, transform= None, mode = 'Vanilla'):
    if mode != 'None':
        model = model.eval()
        class_idx_str = 'class_idx_str'
        #im_orig = LoadImage(img_list)
        Orig_img = Img_DataLoader(img_list= image_list, split='viz',df= df,transform =transform )
        shuffle = False
        dataloader = DataLoader(Orig_img, batch_size=32, num_workers=2, shuffle=shuffle)

        for _data in dataloader:
            break
        im_tensor = _data['image']
        # Show the image

        predictions = model(im_tensor)
        predictions = predictions.detach().numpy()

        prediction_class = np.argmax(predictions, axis=1).tolist()
        
        class_idx_str = 'class_idx_str'

        ROWS = 1
        COLS = number
        UPSCALE_FACTOR = 3

         # Construct the saliency object. This alone doesn't do anthing.


        gradient_saliency = saliency.GradientSaliency()
        integrated_gradients = saliency.IntegratedGradients()
        
    number = len(image_list)
    ROWS = 1
    COLS = number
    UPSCALE_FACTOR = 3
    fig = plt.figure(figsize=(number*3, 3))
    for _order in range(number):
        
        im_orig = LoadImage(image_list[_order])
        im = im_orig.astype(np.float32)
        if mode != 'None':
            baseline = np.zeros(im.shape)
            call_model_args = {class_idx_str: prediction_class}
            call_model_args['model'] = model
        if mode =='Vanilla':
            mask_3d = gradient_saliency.GetMask(im, call_model_function, call_model_args)
        elif mode == 'SmoothGrad':
            mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)
        elif mode == 'Integrated Gradients':
            mask_3d = integrated_gradients.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
        elif mode == "Smooth Integrated Gradients":
            mask_3d = integrated_gradients.GetSmoothedMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
        elif mode == "Guided Integrated Gradients":
            guided_ig = saliency.GuidedIG()
            mask_3d = guided_ig.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)
        elif mode =='XRAI heatmap':
            xrai_object = saliency.XRAI()
            xrai_attributions = xrai_object.GetMask(im, call_model_function, call_model_args, batch_size=20)
        elif mode == 'None':
            pass
            

        else:
            pass
        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        
        if 'mask_3d' in locals():
            smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(mask_3d)
            fig.add_subplot(1, number, _order + 1)
            ShowGrayscaleImage(smoothgrad_mask_grayscale) 
        elif 'xrai_attributions' in locals():
            fig.add_subplot(1, number, _order + 1)
            ShowHeatMap(xrai_attributions)
        else:
            fig.add_subplot(1, number, _order + 1)
            ShowImage(im_orig)