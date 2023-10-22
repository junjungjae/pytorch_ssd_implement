import torch
import json
import os

import numpy as np
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from PIL import Image

import conf as cfg


def make_json_file(vocpath):
    abspath = os.path.abspath(vocpath)

    for data_split in ['train', 'trainval', 'val']:
        with open(f"{abspath}/ImageSets/Main/{data_split}.txt", "r") as f:
            file_path_list = f.read().splitlines()
        
        img_path_list = [f"{abspath}/JPEGImages/{i}.jpg" for i in file_path_list]
        xml_path_list = [f"{abspath}/Annotations/{i}.xml" for i in file_path_list]
        xml_parse_list = []

        for single_xmlpath in xml_path_list:
            tree = ET.parse(single_xmlpath)
            root = tree.getroot()
            bbox_dict = {"boxes": [], "labels": []}

            for _object in root.findall("object"):
                label = cfg.CLASSES_DICT[_object.find("name").text]
                xmin = int(_object.find("bndbox").find("xmin").text) - 1
                ymin = int(_object.find("bndbox").find("ymin").text) - 1
                xmax = int(_object.find("bndbox").find("xmax").text) - 1
                ymax = int(_object.find("bndbox").find("ymax").text) - 1

                bbox_dict['boxes'].append([xmin, ymin, xmax, ymax])
                bbox_dict['labels'].append(label)                
            
            xml_parse_list.append(bbox_dict)

        with open(f"./pre_defined_data/{data_split}_images.json", "w") as f:
            json.dump(img_path_list, f)
        
        with open(f"./pre_defined_data/{data_split}_annot.json", "w") as f:
            json.dump(xml_parse_list, f)

    
class CustomDataset(Dataset):
    def __init__(self, split, apply_transform):
        super(CustomDataset, self).__init__()
        
        with open(f"./{split}_images.json", "r") as f:
            self.images = json.load(f)
        
        with open(f"./{split}_objects.json", "r") as f:
            self.objects = json.load(f)
        
        self.transform = apply_transform

    def __getitem__(self, i):
        image = np.array(Image.open(self.images[i]))

        boxes = self.objects[i]['boxes']
        labels = self.objects[i]['labels']

        augmentation = self.transform(image=image, bboxes=boxes, class_labels=labels)
        image = augmentation['image']
        boxes = torch.tensor(augmentation['bboxes']) / 300
        labels = torch.tensor(augmentation['class_labels'])
        
        return image, boxes, labels

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels
    
if __name__ == "__main__":
    vocpath = "path/to/your/VOC2007"
    make_json_file(vocpath)
    print("Dataset Preprocessing complete.")