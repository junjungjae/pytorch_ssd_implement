import torch

import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_transforms

from torch.utils.data import DataLoader

import utils

import conf as cfg

from custom_dataloader import CustomDataset
from models import SSD300, MultiBoxLoss




train_transform = A.Compose([A.Resize(300, 300),
                             A.RandomCrop(300, 300, p=0.5),
                             A.OpticalDistortion(p=0.5),
                             A.HorizontalFlip(p=0.5),
                             A.Normalize(),
                             A_transforms.ToTensorV2()
                            ],
                             bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

valid_transform = A.Compose([A.Resize(300, 300),
                             A.Normalize(),
                             A_transforms.ToTensorV2()
                             ],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


train_dataset = CustomDataset(split='TRAIN', apply_transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=train_dataset.collate_fn, shuffle=True, pin_memory=True)

valid_dataset = CustomDataset(split='VALID', apply_transform=valid_transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, collate_fn=train_dataset.collate_fn, shuffle=True, pin_memory=True)



lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
epochs = 200

start_epoch = 0
biases = []
not_biases = []

model = SSD300(n_classes=len(cfg.voc_labels) + 1).to(cfg.DEVICE)

for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
            
        else:
            not_biases.append(param)

optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)

criterion = MultiBoxLoss(anchor_boxes=model.anchor_boxes).to(cfg.DEVICE)

best_loss = np.inf


for epoch in range(epochs):        
    train_avg_loss = 0
    valid_avg_loss = 0
    
    model.train()
    for train_minibatch, (train_batch_images, train_batch_bboxes, train_batch_classes) in enumerate(train_dataloader):        
        images = train_batch_images.to(cfg.DEVICE)
        boxes = [b.to(cfg.DEVICE) for b in train_batch_bboxes]
        labels = [l.to(cfg.DEVICE) for l in train_batch_classes]
        
        pred_loc, pred_cls = model(images)
        loss = criterion(pred_loc, pred_cls, boxes, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        utils.clip_gradient(optimizer, 5)
        optimizer.step()
        
        train_avg_loss = (train_avg_loss * train_minibatch + loss.item()) / (train_minibatch + 1)        
        
        print(f"Epoch: {epoch+1}\tBatch: {train_minibatch+1}/{len(train_dataloader)}\tTrain Loss: {train_avg_loss:2f}", end='\r')
    print()
    
    with torch.no_grad():
        model.eval()
        for valid_minibatch, (valid_batch_images, valid_batch_bboxes, valid_batch_classes) in enumerate(valid_dataloader):        
            images = valid_batch_images.to(cfg.DEVICE)
            boxes = [b.to(cfg.DEVICE) for b in valid_batch_bboxes]
            labels = [l.to(cfg.DEVICE) for l in valid_batch_classes]
            
            pred_loc, pred_cls = model(images)
            loss = criterion(pred_loc, pred_cls, boxes, labels)
            
            valid_avg_loss = (valid_avg_loss * valid_minibatch + loss.item()) / (valid_minibatch + 1)        
            
            print(f"Epoch: {epoch+1}\tBatch: {valid_minibatch+1}/{len(valid_dataloader)}\tValid Loss: {valid_avg_loss:2f}", end='\r')
        print("\n")
    
    if valid_avg_loss < best_loss:
        best_loss = valid_avg_loss
        utils.save_checkpoint(epoch, model, optimizer)