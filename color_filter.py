import cv2
import numpy as np

img = cv2.imread('19-maskprojected.jpg')
# img = cv2.imread('examples/eval3-maskprojected.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# yellow
# lower = np.array([20,100,100])
# upper = np.array([30,255,255])

# lower = np.array([45,60,50])
# 55, 30, 60
lower = np.array([26,70,153])
# upper = np.array([162,100,100])
# 72, 100, 100
upper = np.array([36,255,255])

mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(img, img, mask=mask)

# dst = cv2.fastNlMeansDenoisingColored(res,None,50,50,7,21)
# blur = cv2.GaussianBlur(res, (3,3), sigmaX=8, sigmaY=5)

cv2.imshow('img', img)
cv2.imshow('mask',mask)
# cv2.imwrite('examples/embrapa/eval3-yellowFilter-mask.jpg', mask)
cv2.imshow('res',res)
# cv2.imwrite('examples/embrapa/eval3-yellowFilter-res.jpg', res)
# cv2.imshow('dst',dst)
# cv2.imwrite('examples/embrapa/eval3-yellowFilter-resDenoised.jpg', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
