import torch
from torch import nn

import requests
from helper_function import accuracy_fn,pred_and_plot_image
from function import train_test_loop,save_results_txt,Save
from poltFunction import plot_loss_curves


import torchvision
import os
import pathlib
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from typing import Tuple,Dict,List
from torchinfo import summary
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights


#####

device = "cuda" if torch.cuda.is_available else "cpu"

train_dir = "data/photo/train"
test_dir = "data/photo/test"

# prepare dataset
#All pre-trained models expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]
# hal ini bertujuan agar image yang jika miliki bisa meiliki pixel dengan rentang [0,1] dan akan menyesuaikan dengan
# model preTrained

aug_manual_transfroms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

manual_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# loading data

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=aug_manual_transfroms)
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=manual_transforms)

# setup data loader and class name
train_dataLoader = DataLoader(dataset=train_data,
                               batch_size=16,
                               shuffle=True)
test_dataLoader = DataLoader(dataset=test_data,
                             batch_size=16)
class_name = train_data.classes
print(class_name)
# setup weight untuk preTrained model
# hal ini adalah langkah unutk memilih seberapa bagus model yang akan di pakai
# contoh:
# Old weights with accuracy 76.130%
#resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# New weights with accuracy 80.858%
# resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# pre trained model ini sudah di latih dengan menggunakan ImageNet, dan memiliki persentasi yang lebih bagus

# efficiennNet_b0 -> nama model
# IMAGENET1K_V2 -> weight

# setup weight and auto creation
weight = EfficientNet_B0_Weights.IMAGENET1K_V1
# print(weight)
# Get the transforms used to create our pretrained weights
auto_transform = weight.transforms()
# print(auto_transform)

# setup model
model = efficientnet_b0(weight).to(device)

# summaryModel = summary(model=model,
#                        input_size=(16,3,224,244),
#                        col_names=["input_size", "output_size", "num_params", "trainable"],
#                        col_width=20,
#                        row_settings=["var_names"])
# print(summaryModel)

# freezing base model
# melakukan freeze base model dan mengubah output layer sesuai kebutuhan
# Note: To freeze layers means to keep them how they are during training.
    # For example, if your model has pretrained layers, to freeze them would be to say,
    # "don't change any of the patterns in these layers during training, keep them how they are."
    # In essence, we'd like to keep the pretrained weights/patterns our model has learned from ImageNet as a backbone
    # and then only change the output layers.

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False
# Note: Dropout layers randomly remove connections between two neural network layers with a probability of p.
    # For example, if p=0.2, 20% of connections between neural network layers will be removed at random each pass.
    # This practice is meant to help regularize (prevent overfitting) a model by making sure the connections that remain learn features
    # to compensate for the removal of the other connections (hopefully these remaining features are more general).

# set manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# get lem of class
ouput_shape = len(class_name)

# Recreate the classifier layer and seed it to the target device
model.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=ouput_shape,
                    bias=True).to(device)
)
# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
# summaryModel_after_freezing  = summary(model,
#                                        verbose=0,
#                                        input_size=(16,3,224,244),
#                                        col_names=["input_size", "output_size", "num_params", "trainable"],
#                                        col_width=20,
#                                        row_settings=["var_names"])
# print(summaryModel_after_freezing)

# train model
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)

model_results = train_test_loop(epochs=9,
                                model=model,
                                lossFn=loss_fn,
                                optimizer=optimizer,
                                train_dataLoader=train_dataLoader,
                                test_dataLoader=test_dataLoader,
                                perBatch=None)

# plot_loss_curves(model_results)
# plt.show()

test_custom_image_path = "data/test_image/rafa.jpg"

image_custom_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
])

pred_and_plot_image(model=model,
                    image_path=test_custom_image_path,
                    class_names=class_name,
                    transform=image_custom_transform,
                    device=device)
plt.show()