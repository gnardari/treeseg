import cv2
import os

video_dir = '/home/gnardari/Documents/dd/dados/11-06-18/videos'

imsize = 256
videoFile = os.path.join(video_dir, 'GOPR2364.MP4')
imagesFolder = 'frames'
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
cap.set(1, frameRate*187)

stitcher = cv2.createStitcher()
mosaic = None
i = 0
images = []

while(cap.isOpened()):
    msec = cap.get(0) #current frame number
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()

    if not ret:
        continue

    if frameId % 12 == 0:
        # cv2.imshow('aaa', frame)
        # cv2.waitKey()
        frame = cv2.resize(frame, (480,360))
        images.append(frame)
        i+= 1

    if frameId > frameRate*216:
        break

print(len(images))
for i, im in enumerate(images):
    cv2.imwrite('data/mosaic/{}.jpg'.format(i), im)
# r, mosaic = stitcher.stitch(images)
# print(r)
# print(mosaic)
# cv2.imwrite('data/mosaic.jpg', mosaic)
# cap.release()
# print("Done!")
