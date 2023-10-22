import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----- model variable -----
fmap_dims = {'conv4_3': 38,
             'conv7': 19,
             'conv8_2': 10,
             'conv9_2': 5,
             'conv10_2': 3,
             'conv11_2': 1}

obj_scales = {'conv4_3': 0.1,
             'conv7': 0.2,
             'conv8_2': 0.375,
             'conv9_2': 0.55,
             'conv10_2': 0.725,
             'conv11_2': 0.9}

aspect_ratios = {'conv4_3': [1., 2., 0.5],
                 'conv7': [1., 2., 3., 0.5, .333],
                 'conv8_2': [1., 2., 3., 0.5, .333],
                 'conv9_2': [1., 2., 3., 0.5, .333],
                 'conv10_2': [1., 2., 0.5],
                 'conv11_2': [1., 2., 0.5]}

# ----- Dataset Variable -----
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


CLASSES_DICT = {k: v + 1 for v, k in enumerate(voc_labels)}
CLASSES_DICT['background'] = 0
REVERSE_CLASSES_DICT = {v: k for k, v in CLASSES_DICT.items()}

# ----- visualize variable -----
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

label_color_map = {k: distinct_colors[i] for i, k in enumerate(CLASSES_DICT.keys())}