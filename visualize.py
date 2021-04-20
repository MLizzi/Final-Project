# From: https://github.com/jacobgil/pytorch-grad-cam
# Start by running pip install grad-cam
import cv2
import numpy as np
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
import models
import torch

model = models.load_modified_pre_trained('resnet18', 4)

# Place model on device
# model = model.to(device)

# parameters to adjust for different results to be generated
imagePath = r'D:\Documents\4th Year\CSC413\Final-Project\sample_test_images'
resultsPath = r'D:\Documents\4th Year\CSC413\Final-Project'
model_filenames = ['new_resnet18_30e_overlap_model', 'new_resnet18_30e_model']
imageNames = ['test_boar.jpg']#['test_rodent.jpg', 'test_puma_1.jpg', 'test_puma_2.jpg', 'test_rodent.jpg', 'test_turkey.jpg']  # swap around based on image wanted

# Load the weights
for model_filename in model_filenames:
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    # model = resnet50(pretrained=True) # TO SWAP OUT WITH OUR MODEL
    target_layer1 = model.layer1[-1] # TO ADJUST IF OUR MODEL HAS MORE LAYERS (I.E. OUTPUT LAYER NUMBER IS DIFFERENT)
    target_layer2 = model.layer2[-1] # TO ADJUST IF OUR MODEL HAS MORE LAYERS (I.E. OUTPUT LAYER NUMBER IS DIFFERENT)
    target_layer3 = model.layer3[-1] # TO ADJUST IF OUR MODEL HAS MORE LAYERS (I.E. OUTPUT LAYER NUMBER IS DIFFERENT)
    target_layer4 = model.layer4[-1] # TO ADJUST IF OUR MODEL HAS MORE LAYERS (I.E. OUTPUT LAYER NUMBER IS DIFFERENT)
    target_layers = [target_layer1, target_layer2, target_layer3, target_layer4]

    for imageName in imageNames:
        rgb_img = cv2.imread(os.path.join(imagePath, imageName), 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #input_tensor = # ADD IMAGE TENSOR (either through

        #Can be GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
        for i, layer in enumerate(target_layers):
            cam = GradCAM(model=model, target_layer=layer, use_cuda=False)
            grayscale_cam = cam(input_tensor=input_tensor, target_category=0)
            visualization = show_cam_on_image(rgb_img, grayscale_cam)

            # cv2.imshow('GradCAM ' + imageName.replace('.jpg',''), visualization)
            # cv2.waitKey(500) #Wait 1/2 a second before cycling to next photo

            folder = 'GradCAM_{}_{}'.format(imageName.replace('.jpg',''), model_filename)
            if not os.path.exists(folder):
                os.makedirs(folder)

            print(cv2.imwrite(os.path.join(resultsPath, folder, 'GradCAM_{}_{}_layer{}.jpg'.format(imageName.replace('.jpg',''),model_filename, i+1)), visualization))
            # cv2.waitKey(250)# cv2.waitKey(0)