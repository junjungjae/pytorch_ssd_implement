import torch

import torch.nn as nn
import torch.nn.functional as F
import conf as cfg
import numpy as np

from torchvision.models import vgg16
from torchvision.ops import box_convert, box_iou
from utils import cxcywh_to_gcxgcy, gcxgcy_to_cxcywh

class VGG16_base(nn.Module):
    """
    SSD에서 feature extraction을 위해 사용되는 base network.
    """
    def __init__(self):
        super(VGG16_base, self).__init__()
        
        # 논문 구조상 fully-connected layer을 대체한 conv layer 사전 정의
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        # base network에서 사용할 base network의 feature
        self.conv4_3_net = None
        self.conv6_net = None
        
        self.get_base_network()
        
    def get_base_network(self):
        ceil_pooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        re_pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        base = nn.Sequential(*vgg16(weights=True).children())[0]
        base[16] = ceil_pooling
        base[30] = re_pool5
        
        # 빠른 학습을 위해 weight freezing
        for param in base.parameters():
            param.requires_grad = False
        
        self.conv4_3_net = base[:21]
        self.conv6_net = base
    
    def forward(self, x):
        conv4_3_features = self.conv4_3_net(x)
        
        out = self.conv6_net(x)        
        out = F.relu(self.conv6(out))
        conv7_feats = F.relu(self.conv7(out))

        return conv4_3_features, conv7_feats
        

class AuxiliaryConvolutions(nn.Module):
    """
    base network 이후에 오는 보조 네트워크.
    base network의 feature(conv4_3, conv7)와 보조 네트워크의 feature(conv8_2, conv9_2, conv10_2, conv11_2)를 기반으로
    anchor box 생성
    """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):        
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_feats = out

        out = F.relu(self.conv11_1(out))
        conv11_2_feats = F.relu(self.conv11_2(out))

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):
    """
    각 feature의 point들에 대해 anchor box 생성, box regression, classification을 수행하는 네트워크
    """
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # 사전 정의된 feature 별 anchor box 생성 개수. 
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}

        # 각 feature 별 box regression layer
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # 각 feature 별 box classification layer
        self.cls_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cls_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cls_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cls_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cls_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cls_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        # 입력받은 feature의 shape은 (batch_size, channel, feature_h, feature_w)의 형태를 가짐
        # 초기 정의한 regression, classification layer 연산을 위해 차원 변경
        # (batch_size, channel, feature_h, feature_w) -> (batch_size, feature_h, feature_w, channel)
        # contiguous의 경우 permute, view, reshape 등의 차원 변형 시 발생할 수 있는 메모리 할당 문제를 해결하기 위함
        # -> 메모리상에 불연속적으로 저장된 일련의 텐서들을 연속적으로 저장되게 함으로서 forward, back propagation, GPU 연산 시 속도적 이점을 얻을 수 있음.
        loc_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        loc_conv4_3 = loc_conv4_3.permute(0, 2, 3, 1).contiguous()
        loc_conv4_3 = loc_conv4_3.view(batch_size, -1, 4)

        loc_conv7 = self.loc_conv7(conv7_feats)
        loc_conv7 = loc_conv7.permute(0, 2, 3, 1).contiguous()
        loc_conv7 = loc_conv7.view(batch_size, -1, 4)

        loc_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        loc_conv8_2 = loc_conv8_2.permute(0, 2, 3, 1).contiguous()
        loc_conv8_2 = loc_conv8_2.view(batch_size, -1, 4)

        loc_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        loc_conv9_2 = loc_conv9_2.permute(0, 2, 3, 1).contiguous()
        loc_conv9_2 = loc_conv9_2.view(batch_size, -1, 4)

        loc_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        loc_conv10_2 = loc_conv10_2.permute(0, 2, 3, 1).contiguous()
        loc_conv10_2 = loc_conv10_2.view(batch_size, -1, 4)

        loc_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        loc_conv11_2 = loc_conv11_2.permute(0, 2, 3, 1).contiguous()
        loc_conv11_2 = loc_conv11_2.view(batch_size, -1, 4)

        cls_conv4_3 = self.cls_conv4_3(conv4_3_feats)
        cls_conv4_3 = cls_conv4_3.permute(0, 2, 3, 1).contiguous()
        cls_conv4_3 = cls_conv4_3.view(batch_size, -1, self.n_classes)

        cls_conv7 = self.cls_conv7(conv7_feats)
        cls_conv7 = cls_conv7.permute(0, 2, 3, 1).contiguous()
        cls_conv7 = cls_conv7.view(batch_size, -1, self.n_classes)

        cls_conv8_2 = self.cls_conv8_2(conv8_2_feats)
        cls_conv8_2 = cls_conv8_2.permute(0, 2, 3, 1).contiguous()
        cls_conv8_2 = cls_conv8_2.view(batch_size, -1, self.n_classes)

        cls_conv9_2 = self.cls_conv9_2(conv9_2_feats)
        cls_conv9_2 = cls_conv9_2.permute(0, 2, 3, 1).contiguous()
        cls_conv9_2 = cls_conv9_2.view(batch_size, -1, self.n_classes)

        cls_conv10_2 = self.cls_conv10_2(conv10_2_feats)
        cls_conv10_2 = cls_conv10_2.permute(0, 2, 3, 1).contiguous()
        cls_conv10_2 = cls_conv10_2.view(batch_size, -1, self.n_classes)

        cls_conv11_2 = self.cls_conv11_2(conv11_2_feats)
        cls_conv11_2 = cls_conv11_2.permute(0, 2, 3, 1).contiguous()
        cls_conv11_2 = cls_conv11_2.view(batch_size, -1, self.n_classes)

        locs = torch.cat([loc_conv4_3, loc_conv7, loc_conv8_2, loc_conv9_2, loc_conv10_2, loc_conv11_2], dim=1)
        classes_scores = torch.cat([cls_conv4_3, cls_conv7, cls_conv8_2, cls_conv9_2, cls_conv10_2, cls_conv11_2], dim=1)

        return locs, classes_scores
    
    
class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        
        self.n_classes = n_classes
        
        self.base_network = VGG16_base()
        self.auxiliary = AuxiliaryConvolutions()
        self.prediction = PredictionConvolutions(n_classes=n_classes)
        
        self.anchor_boxes = self.generate_anchor_box()
    
    
    def generate_anchor_box(self):
        """
        각 feature map에 대한 anchor box 생성
        각 anchor box의 width, height는 기 정의된 scale, aspect ratio를 사용
        논문 구조에 따라 모든 anchor box들의 중심 좌표는 0~1로 rescale
        """
        feature_map_layer_list = list(cfg.fmap_dims.keys())        
        anchor_boxes = []

        for k, fmap in enumerate(feature_map_layer_list):
            for i in range(cfg.fmap_dims[fmap]):
                for j in range(cfg.fmap_dims[fmap]):
                    cx = (j + 0.5) / cfg.fmap_dims[fmap]
                    cy = (i + 0.5) / cfg.fmap_dims[fmap]

                    for ratio in cfg.aspect_ratios[fmap]:
                        anchor_boxes.append([cx, cy, cfg.obj_scales[fmap] * np.sqrt(ratio), cfg.obj_scales[fmap] / np.sqrt(ratio)])

                        # 기존 정의된 scale 및 aspect ratio에 더해, 모델에 robustness를 더해주기 위해
                        # aspect ratio가 1일 경우 현재, 다음 feature map size의 기하평균 만큼을 새로운 scale로 정의하여 anchor box 추가
                        if ratio == 1.:
                            try:
                                additional_scale = np.sqrt(cfg.obj_scales[fmap] * cfg.obj_scales[feature_map_layer_list[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            anchor_boxes.append([cx, cy, additional_scale, additional_scale])

        anchor_boxes = torch.FloatTensor(anchor_boxes).to(cfg.DEVICE)
        anchor_boxes.clamp_(0, 1)
        
        return anchor_boxes

    
    def forward(self, x):
        x = x.float()
        conv4_3_features, conv7_features = self.base_network(x)
        conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features = self.auxiliary(conv7_features)
        locations, classes_score = self.prediction(conv4_3_features, conv7_features, conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features)
        
        return locations, classes_score
    
    def inference(self, predicted_locs, predicted_scores, min_score, max_overlap, top_n):
        """
        model inference 파트
        min_score: confidence score 최소값.
        max_overlap: NMS 수행 시 overlap 허용 최대값
        top_k: confidence 기준 상위 k개의 결과만 표출
        """
        batch_size = predicted_locs.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        for i in range(batch_size):
            decoded_locs = box_convert(gcxgcy_to_cxcywh(predicted_locs[i], self.anchor_boxes), in_fmt='cxcywh', out_fmt='xyxy')

            image_boxes = []
            image_labels = []
            image_scores = []

            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                
                if n_above_min_score == 0:
                    continue
                
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                overlap = box_iou(class_decoded_locs, class_decoded_locs)
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(cfg.DEVICE)

                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue

                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(cfg.DEVICE))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                # 적절한 결과가 없을 경우
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(cfg.DEVICE))
                image_labels.append(torch.LongTensor([0]).to(cfg.DEVICE))
                image_scores.append(torch.FloatTensor([0.]).to(cfg.DEVICE))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_objects = image_scores.size(0)

            if n_objects > top_n:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_n]
                image_boxes = image_boxes[sort_ind][:top_n]
                image_labels = image_labels[sort_ind][:top_n]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores
    
    

class MultiBoxLoss(nn.Module):
    def __init__(self, anchor_boxes, iou_threshold=0.5, negative_mining_ratio=3, alpha=1):
        super(MultiBoxLoss, self).__init__()
        
        self.anchor_boxes = anchor_boxes
        self.iou_threshold = iou_threshold
        self.negative_mining_ratio = negative_mining_ratio
        self.alpha = alpha
        
        self.cls_loss = nn.CrossEntropyLoss(reduce=False)
        self.reg_loss = nn.SmoothL1Loss()
    
    def forward(self, pred_locs, pred_cls, bboxes, labels):
        batch_size = pred_locs.size(0)
        n_anchors = self.anchor_boxes.size(0)
        n_classes = pred_cls.size(2)
        
        gt_bboxes = torch.zeros((batch_size, n_anchors, 4), dtype=torch.float).to(cfg.DEVICE)
        gt_classes = torch.zeros((batch_size, n_anchors), dtype=torch.long).to(cfg.DEVICE)
        
        # 각 배치별로 계산
        for i in range(batch_size):
            objects_n = bboxes[i].size(0)
            iou = box_iou(bboxes[i], self.anchor_boxes)
            iou_val, iou_ind = iou.max(dim=0)
            
            # target과 pred box간 iou 계산
            # 최대 iou값에 대한 indices 추출 후 해당 박스들에 대해서만 계산
            _, max_iou_ind = iou.max(dim=1)
            iou_ind[max_iou_ind] = torch.LongTensor(range(objects_n)).to(cfg.DEVICE)
            iou_val[max_iou_ind] = 1
            
            anchor_boxes_label = labels[i][iou_ind]
            anchor_boxes_label[iou_val < self.iou_threshold] = 0
            
            gt_classes[i] = anchor_boxes_label
            gt_bboxes[i] = cxcywh_to_gcxgcy(box_convert(bboxes[i][iou_ind], in_fmt='xyxy', out_fmt='cxcywh'), self.anchor_boxes)
            
        positive_anchor_boxes = gt_classes != 0  # positive anchor box mask
        loc_loss = self.reg_loss(pred_locs[positive_anchor_boxes], gt_bboxes[positive_anchor_boxes])
        
        positive_n = positive_anchor_boxes.sum(dim=1)
        negative_n = self.negative_mining_ratio * positive_n
        
        conf_loss_all = self.cls_loss(pred_cls.view(-1, n_classes), gt_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, -1)
        
        conf_loss_positive = conf_loss_all[positive_anchor_boxes]
        
        conf_loss_negative = conf_loss_all.clone()
        conf_loss_negative[positive_anchor_boxes] = 0
        conf_loss_negative, _ = conf_loss_negative.sort(dim=1, descending=True)
        
        hard_negative_ind = torch.LongTensor(range(n_anchors)).unsqueeze(0).expand_as(conf_loss_negative).to(cfg.DEVICE)
        hard_negative_boxes = hard_negative_ind < negative_n.unsqueeze(1)
        conf_loss_hard_negative = conf_loss_negative[hard_negative_boxes]
        
        conf_loss = (conf_loss_hard_negative.sum() + conf_loss_positive.sum())  / positive_n.sum().float()
        
        return conf_loss + self.alpha * loc_loss
    