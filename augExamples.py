from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

IMG_SHAPE = (512,512,3)
BATCH_SIZE = 8

blur = iaa.GaussianBlur(1.25)
multiply = iaa.Multiply((0.8, 1.2))

def augment(img, soof):
    return soof.augment_image(img*255)

img = cv2.cvtColor(cv2.imread('eval3.jpg'), cv2.COLOR_BGR2RGB)*(1./255)
augImg = augment(img, multiply)
cv2.imwrite('eval3Dark.jpg', cv2.cvtColor(augImg, cv2.COLOR_RGB2BGR))

fig = plt.figure()
fig.add_subplot(1,2,1).title.set_text('Original')
plt.imshow(img)

fig.add_subplot(1,2,2).title.set_text('Aug')
plt.imshow(augImg)
plt.show()
