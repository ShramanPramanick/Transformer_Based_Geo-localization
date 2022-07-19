from dataset import MultiPartitioningClassifier, cuda_base, device_ids, scenes, num_epochs
import yaml
from argparse import Namespace
import torch
import argparse

with open('../config/base_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

train_data_loader = tmp_model.train_dataloader()
val_data_loader = tmp_model.val_dataloader()
# Choose the first n_steps batches with 64 samples in each batch
# n_steps = 10

import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, confusion_matrix
from torchsummary import summary
import random
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from torchsummary import summary
from transformers import ViTModel
import warnings
import math
warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

def topk_accuracy(target, output, k):
    topn = np.argsort(output, axis = 1)[:,-k:]
    return np.mean(np.array([1 if target[k] in topn[k] else 0 for k in range(len(topn))]))


def adjust_learning_rate(num_epochs, optimizer, loader, step):
    max_steps = num_epochs * len(loader)
    warmup_steps = 2 * len(loader) ### In originalBT repo, this constant is 10
    base_lr = 0.1
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * 0.2
    optimizer.param_groups[1]['lr'] = lr * 0.0048


num_classes_coarse = 3298
num_classes_middle = 7202
num_classes_fine = 12893
learning_rate = config["lr"]
num_scene = int(scenes)

class GeoClassification(nn.Module):

    def __init__(self):
        super(GeoClassification, self).__init__()
        
        self.vit = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        self.classifier_1 = nn.Linear(self.vit.config.hidden_size, num_classes_coarse)
        self.classifier_2 = nn.Linear(self.vit.config.hidden_size, num_classes_middle)
        self.classifier_3 = nn.Linear(self.vit.config.hidden_size, num_classes_fine)
        self.classifier_4 = nn.Linear(self.vit.config.hidden_size, num_scene)
    
    def forward(self, rgb_image):
        
        outputs = self.vit(rgb_image).last_hidden_state
        outputs = outputs[:,0,:]
        logits_geocell_coarse = self.classifier_1(outputs)
        logits_geocell_middle = self.classifier_2(outputs)
        logits_geocell_fine = self.classifier_3(outputs)
        logits_scene = self.classifier_2(outputs)
        return logits_geocell_coarse, logits_geocell_middle, logits_geocell_fine, logits_scene

device = torch.device(cuda_base if torch.cuda.is_available() else 'cpu')
model = GeoClassification()     
model = model.to(device)

param_weights = []
param_biases = []
for param in model.parameters():
    if param.ndim == 1:
        param_biases.append(param)
    else:
        param_weights.append(param)
parameters = [{'params': param_weights}, {'params': param_biases}]


model = nn.DataParallel(model, device_ids=device_ids)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum = config['momentum'], weight_decay = config["weight_decay"])
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = config["milestones"], gamma= config["gamma"])

#print(summary(model, (3, 224, 224)))

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []

import warnings
warnings.filterwarnings("ignore")

n_total_steps = len(train_data_loader)

batch_wise_loss = []
batch_wise_micro_f1 = []
batch_wise_macro_f1 = []
epoch_wise_top_1_accuracy = []
epoch_wise_top_10_accuracy = []
epoch_wise_top_50_accuracy = []
epoch_wise_top_100_accuracy = []
epoch_wise_top_200_accuracy = []
epoch_wise_top_300_accuracy = []
epoch_wise_top_500_accuracy = []


scaler = torch.cuda.amp.GradScaler()
for epoch in range(num_epochs):
    steps = len(train_data_loader) * epoch
    for i, (rgb_image, _, label, _, _, scene) in enumerate(train_data_loader):
        
        rgb_image = rgb_image.type(torch.float32).to(device)
        label_coarse = label[0].to(device)
        label_middle = label[1].to(device)
        label_fine = label[2].to(device)
        scene = scene.type(torch.long).to(device)

        adjust_learning_rate(num_epochs, optimizer, train_data_loader, steps)
        optimizer.zero_grad()
        steps += 1
        
         # Forward pass
        model.train()
        batch_size, n_crops, c, h, w = rgb_image.size()

        with torch.cuda.amp.autocast():
            outputs_geocell_coarse, outputs_geocell_middle, outputs_geocell_fine, outputs_scene =  model(rgb_image.view(-1, c, h, w))
            outputs_geocell_coarse = outputs_geocell_coarse.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_middle = outputs_geocell_middle.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_fine = outputs_geocell_fine.view(batch_size, n_crops, -1).mean(1)
            outputs_scene = outputs_scene.view(batch_size, n_crops, -1).mean(1)
            loss_geocell_coarse = criterion(outputs_geocell_coarse, label_coarse)
            loss_geocell_middle = criterion(outputs_geocell_middle, label_middle)
            loss_geocell_fine = criterion(outputs_geocell_fine, label_fine)
            loss_scene = criterion(outputs_scene, scene)

            loss = 0.5*loss_geocell_coarse + 0.3*loss_geocell_middle + 0.2*loss_geocell_fine + loss_scene
        
        # Backward and optimize

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (i+1) % 4000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        #if (i+1) == n_steps:
            #break

    target_total_test = []
    predicted_total_test = []
    model_outputs_total_test = []

    with torch.no_grad():
        
        n_correct = 0
        n_samples = 0

        for i, (rgb_image, _, label, _, _, scene) in enumerate(val_data_loader):
            
            rgb_image = rgb_image.type(torch.float32).to(device)

            label_fine = label[2].to(device)
            scene = scene.type(torch.long).to(device)

            # Forward pass
            model.eval()
            batch_size, n_crops, c, h, w = rgb_image.size()
            outputs_geocell_coarse, outputs_geocell_middle, outputs_geocell_fine, outputs_scene =  model(rgb_image.view(-1, c, h, w))
            outputs_geocell_coarse = outputs_geocell_coarse.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_middle = outputs_geocell_middle.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_fine = outputs_geocell_fine.view(batch_size, n_crops, -1).mean(1)
            outputs_scene = outputs_scene.view(batch_size, n_crops, -1).mean(1)
            #print(outputs)
            # max returns (value ,index)
            _, predicted = torch.max(outputs_geocell_fine.data, 1)
            #print(label)
            #print(predicted)
            n_samples += label_fine.size(0)
            n_correct += (predicted == label_fine).sum().item()

            target_total_test.append(label_fine)
            predicted_total_test.append(predicted)
            model_outputs_total_test.append(outputs_geocell_fine)

            target_inter = [t.cpu().numpy() for t in target_total_test]
            predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
            outputs_inter = [t.cpu().numpy() for t in model_outputs_total_test]
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            outputs_inter = np.concatenate(outputs_inter, axis=0)
        
        current_top_1_accuracy = topk_accuracy(target_inter, outputs_inter, k=1)
        epoch_wise_top_1_accuracy.append(current_top_1_accuracy)
        current_top_10_accuracy = topk_accuracy(target_inter, outputs_inter, k=10)
        epoch_wise_top_10_accuracy.append(current_top_10_accuracy)
        current_top_50_accuracy = topk_accuracy(target_inter, outputs_inter, k=50)
        epoch_wise_top_50_accuracy.append(current_top_50_accuracy)
        current_top_100_accuracy = topk_accuracy(target_inter, outputs_inter, k=100)
        epoch_wise_top_100_accuracy.append(current_top_100_accuracy)
        current_top_200_accuracy = topk_accuracy(target_inter, outputs_inter, k=200)
        epoch_wise_top_200_accuracy.append(current_top_200_accuracy)
        current_top_300_accuracy = topk_accuracy(target_inter, outputs_inter, k=300)
        epoch_wise_top_300_accuracy.append(current_top_300_accuracy)
        current_top_500_accuracy = topk_accuracy(target_inter, outputs_inter, k=500)
        epoch_wise_top_500_accuracy.append(current_top_500_accuracy)
       
        print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
        print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
        print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
        print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
        
        print(f' Best Top_1_accuracy on test set till this epoch: {max(epoch_wise_top_1_accuracy)} Found in Epoch No: {epoch_wise_top_1_accuracy.index(max(epoch_wise_top_1_accuracy))+1}')
        print(f' Best Top_10_accuracy on test set till this epoch: {max(epoch_wise_top_10_accuracy)} Found in Epoch No: {epoch_wise_top_10_accuracy.index(max(epoch_wise_top_10_accuracy))+1}')
        print(f' Best Top_50_accuracy on test set till this epoch: {max(epoch_wise_top_50_accuracy)} Found in Epoch No: {epoch_wise_top_50_accuracy.index(max(epoch_wise_top_50_accuracy))+1}')
        print(f' Best Top_100_accuracy on test set till this epoch: {max(epoch_wise_top_100_accuracy)} Found in Epoch No: {epoch_wise_top_100_accuracy.index(max(epoch_wise_top_100_accuracy))+1}')
        print(f' Best Top_200_accuracy on test set till this epoch: {max(epoch_wise_top_200_accuracy)} Found in Epoch No: {epoch_wise_top_200_accuracy.index(max(epoch_wise_top_200_accuracy))+1}')
        print(f' Best Top_300_accuracy on test set till this epoch: {max(epoch_wise_top_300_accuracy)} Found in Epoch No: {epoch_wise_top_300_accuracy.index(max(epoch_wise_top_300_accuracy))+1}')
        print(f' Best Top_500_accuracy on test set till this epoch: {max(epoch_wise_top_500_accuracy)} Found in Epoch No: {epoch_wise_top_500_accuracy.index(max(epoch_wise_top_500_accuracy))+1}')
        print(f' Top_1_accuracy: {epoch_wise_top_1_accuracy}')
        #print(f' Top_500_accuracy: {epoch_wise_top_500_accuracy}')

        if not os.path.exists('../saved_models'):
            os.makedirs('../saved_models')

        if(current_top_1_accuracy == max(epoch_wise_top_1_accuracy)):
            torch.save({'Model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, '../saved_models/ViT_RGB_FourTask.tar')


print("======================================")
print("Training Completed, Evaluating the test set using the best model")
print("======================================")


#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
saved_model = torch.load('../saved_models/ViT_RGB_FourTask.tar')
model.load_state_dict(saved_model['Model_state_dict'])
optimizer.load_state_dict(saved_model['optimizer_state_dict'])

model.to(device)

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []


with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for i, (rgb_image, _, label, _, _, _) in enumerate(val_data_loader):

            rgb_image = rgb_image.type(torch.float32).to(device)

            label_fine = label[2].to(device)
            scene = scene.type(torch.long).to(device)

            # Forward pass
            model.eval()
            batch_size, n_crops, c, h, w = rgb_image.size()
            outputs_geocell_coarse, outputs_geocell_middle, outputs_geocell_fine, outputs_scene =  model(rgb_image.view(-1, c, h, w))
            outputs_geocell_coarse = outputs_geocell_coarse.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_middle = outputs_geocell_middle.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_fine = outputs_geocell_fine.view(batch_size, n_crops, -1).mean(1)
            outputs_scene = outputs_scene.view(batch_size, n_crops, -1).mean(1)
            #print(outputs)
            # max returns (value ,index)
            _, predicted = torch.max(outputs_geocell_fine.data, 1)
            #print(label)
            #print(predicted)
            n_samples += label_fine.size(0)
            n_correct += (predicted == label_fine).sum().item()

            target_total_test.append(label_fine)
            predicted_total_test.append(predicted)
            model_outputs_total_test.append(outputs_geocell_fine)

            target_inter = [t.cpu().numpy() for t in target_total_test]
            predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
            outputs_inter = [t.cpu().numpy() for t in model_outputs_total_test]
            #print(target_inter[-1].shape)
            #print(predicted_inter[-1].shape)
            #print(outputs_inter[-1].shape)
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            outputs_inter = np.concatenate(outputs_inter, axis=0)

        print(f' Accuracy of the network on the test set with the saved model is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
        print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
        print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
        print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
