import argparse

import albumentations as A
import albumentations.pytorch.transforms as A_transforms
import numpy as np

from utils import *
from PIL import ImageDraw, ImageFont, Image

import conf as cfg


def detect(original_image, transfer_module, min_score=0.25, max_overlap=0.5, top_n=1000):
    aug_array = np.array(original_image)
    image = transfer_module(image=aug_array)['image']
    image = image.to(cfg.DEVICE)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    det_boxes, det_labels, _ = model.inference(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_n=top_n)

    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = [cfg.REVERSE_CLASSES_DICT[l+1] for l in det_labels[0].to('cpu').tolist()]
    print(det_labels)

    if det_labels == ['background']:
        return original_image

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    for i in range(det_boxes.size(0)):
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location)
        draw.rectangle(xy=[l + 1. for l in box_location])

        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location)
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="img_path")
    parser.add_argument("--weights", dest="weights_path")
    args = parser.parse_args()
    
    checkpoint = torch.load(args.weights_path)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

    model = checkpoint['model']
    model = model.to(cfg.DEVICE)
    model.eval()


    A_inference = A.Compose([A.Resize(300, 300),
                        A.Normalize(),
                        A_transforms.ToTensorV2()
                        ])
    
    pil_image = Image.open(args.img_path)
    detect(pil_image, transfer_module=A_inference).show()