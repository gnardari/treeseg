import cv2
import os

for img_name in os.listdir('masks/.'):
    if img_name.startswith('.'):
        continue

    img = cv2.imread('masks/' + img_name, 0)
    im_bw = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(os.path.join('bwmasks', img_name), im_bw)
