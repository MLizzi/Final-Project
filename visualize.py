# From: https://github.com/jacobgil/pytorch-grad-cam
# Start by running pip install grad-cam
import cv2
import numpy as np
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50

model = resnet50(pretrained=True) # TO SWAP OUT WITH OUR MODEL
target_layer = model.layer4[-1] # TO ADJUST IF OUR MODEL HAS MORE LAYERS (I.E. OUTPUT LAYER NUMBER IS DIFFERENT)

imagePath = 'D:\Desktop'
imageName = 'dogs.png' # swap around based on image wanted


rgb_img = cv2.imread(os.path.join(imagePath, imageName), 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img)#, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#input_tensor = # ADD IMAGE TENSOR (either through

#Can be GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=False)
grayscale_cam = cam(input_tensor=input_tensor, target_category=1)
visualization = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imshow('GradCAM++ ' + imageName, visualization)
cv2.imwrite(os.path.join(imagePath, 'GradCAM++ ' + imageName), visualization) 
cv2.waitKey(0)