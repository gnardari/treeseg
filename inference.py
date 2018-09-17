from architectures.unet import Unet
from architectures.fcn import FCN
from architectures.deeplab import Deeplabv3
import matplotlib.pyplot as plt
import numpy as np
import cv2

from timeit import default_timer as timer

def expand_channels(img):
    return cv2.merge((img, img, img))

# IMG_SHAPE = (1250,565,3)
IMG_SHAPE = (512,512,3)


model = Unet(input_shape=IMG_SHAPE)
# model = FCN(input_shape=IMG_SHAPE)
# model = Deeplabv3(input_shape=IMG_SHAPE)

# model.load_weights('models/allnew-augmented-fold-unetaug-fold.h5')
# model.load_weights('models/allnew-augmented-fold-fcnaug-fold.h5')
img = cv2.cvtColor(cv2.imread('examples/eval1.jpg'), cv2.COLOR_BGR2RGB)*(1./255)

tt = []
for _ in range(11):
    start = timer()
    res = model.predict(np.expand_dims(img, axis=0), batch_size=1)[0]
    end = timer()
    tt.append(end - start)

print(tt)
print(np.mean(tt[1:]))

# mask = expand_channels(res)
#
# mask = cv2.resize(mask, (565,1250))
#
# img[mask < 0.5] = 0.0
#
# mask[mask > 0.5] = 1.0
# mask[mask < 0.5] = 0.0
#
# def keep_range(m):
#     m[m > 255] = 255
#     m[m < 0] = 0
#     return m.astype(np.uint8)
#
# cv2.imwrite('mosaico-trator-maskprojected.jpg', cv2.cvtColor(keep_range(img*255), cv2.COLOR_RGB2BGR))
# fig = plt.figure()
# fig.add_subplot(1,2,1).title.set_text('Original')
# plt.imshow(img)
#
# fig.add_subplot(1,2,2).title.set_text('Inf')
# plt.imshow(img)
# plt.imshow(mask, alpha=0.7, cmap='RdYlGn')
# plt.show()
