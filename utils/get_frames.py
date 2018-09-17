import cv2
import os

video_dir = '/home/gnardari/Documents/dd/dados/11-06-18/videos'

imsize = 256
videoFile = os.path.join(video_dir, 'GOPR2367.MP4')
imagesFolder = 'frames'
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate

while(cap.isOpened()):
    msec = cap.get(0) #current frame number
    print(msec)
    frameId = cap.get(1) #current frame number

    ret, frame = cap.read()
    if not ret:
        continue

    if (frameId % int(frameRate) == 0):
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        h, w, _ = frame.shape
        cv2.imwrite(filename,
                    frame[(h/2)-imsize:(h/2)+imsize,
                          (w/2)-imsize:(w/2)+imsize, :])

cap.release()
print "Done!"
