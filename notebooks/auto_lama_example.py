import numpy as np
#import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns): #anns is for Artificial Neural Network
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)#highest are will be sorted first
    ax = plt.gca() #Get the current Axes
    ax.set_autoscale_on(False)

    #np.ones will return a new array of given shape and type, filled with ones.
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.50]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('images/Lama.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Orignal image will print
plt.figure(figsize=(20,20))
#plt.imshow(image)
plt.axis('off')
#plt.show()

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
