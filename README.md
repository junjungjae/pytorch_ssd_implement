# pytorch-ssd-implementation

### 공식 논문
LIU, Wei, et al. Ssd: Single shot multibox detector. In: _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14_. Springer International Publishing, 2016. p. 21-37.

### 참고자료
	- https://github.com/amdegroot/ssd.pytorch
	- https://velog.io/@choonsik_mom/SSDSingle-Shot-Multibox-Detector%EB%A1%9C-%EB%AC%BC%EC%B2%B4-%EA%B0%90%EC%A7%80-%EA%B5%AC%ED%98%84
	- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
	- https://github.com/biyoml/PyTorch-SSD

### 데이터 준비
- python custom_dataloader.py --dataset path/to/your/voc2007/folder
	
- ex) python custom_dataloader.py --dataset ./voc/VOCdevkit/VOC2007

### 학습
	- python train.py

### Inference
- python detect.py --source path/of/image --weights path/of/weights