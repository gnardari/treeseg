from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, CSVLogger
from architectures.unet import Unet
from architectures.fcn import FCN
from architectures.deeplab import Deeplabv3
from sklearn.metrics import jaccard_similarity_score as jaccard
from sklearn.metrics import f1_score as f1
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2

IMG_SHAPE = (512,512,3)
BATCH_SIZE = 8
VAL_BATCH_SIZE = 16


# base_dir = '/home/gnardari/treeseg'
base_dir = '.'

# datagen_args = dict(rescale=1./255,
#                     rotation_range=180,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     zoom_range=0.05,
#                     vertical_flip=True,
#                     horizontal_flip=True,
#                     validation_split=0.2)

datagen_args = dict(rescale=1./255, validation_split=0.2)

class ValCallback(Callback):
    def __init__(self, val_gen, batch_size, fold):
        self.val_gen = val_gen
        self.batch_size = batch_size
        self.fold = fold

    def flat_and_binary(self, m):
        b,w,h,c = m.shape
        m = m.reshape((b, w*h*c))

        m[m > 0.5] = 1
        m[m < 0.5] = 0
        return m.astype(int)

    def on_train_end(self, epoch, logs={}):
        x, y = next(self.val_gen)
        preds = self.model.predict(x, batch_size=self.batch_size)

        y = self.flat_and_binary(y)
        preds = self.flat_and_binary(preds)
        jac = jaccard(y, preds)
        dice = f1(y, preds, average='micro')

        logs['dice'] = dice
        logs['jac'] = jac
        logs['fold'] = self.fold

        print('\nTesting jac: {}, dice: {}\n'.format(jac, dice))

def train(model, train_gen, steps, epochs, val_gen, val_steps, fold, model_name):
    # print(model.summary())
    history = model.fit_generator(
                train_gen,
                steps_per_epoch=steps,
                epochs=epochs,
                # validation_data=val_gen,
                # validation_steps=val_steps,
                callbacks=[ValCallback(val_gen, VAL_BATCH_SIZE, fold), CSVLogger(model_name+'.log', append=True)])
    return model, history

def merge_h(h1, h2):
    for k in h1.keys():
        h1[k].extend(h2[k])
    return h1

# img_datagen = ImageDataGenerator(preprocessing_function=augment,
#                                  **datagen_args)

def get_gens(img_datagen, mask_datagen, seed):
    train_img_generator = img_datagen.flow_from_directory(os.path.join(base_dir, 'data/newdata'),
                                                    batch_size=BATCH_SIZE,
                                                    target_size=IMG_SHAPE[:2],
                                                    subset='training',
                                                    class_mode=None, seed=seed)

    val_img_generator = img_datagen.flow_from_directory(os.path.join(base_dir, 'data/newdata'),
                                                    batch_size=VAL_BATCH_SIZE,
                                                    target_size=IMG_SHAPE[:2],
                                                    subset='validation',
                                                    class_mode=None, seed=seed)

    train_mask_generator = mask_datagen.flow_from_directory(os.path.join(base_dir, 'data/newmask'),
                                                      color_mode='grayscale',
                                                      batch_size=BATCH_SIZE,
                                                      target_size=IMG_SHAPE[:2],
                                                      subset='training',
                                                      class_mode=None, seed=seed)

    val_mask_generator = mask_datagen.flow_from_directory(os.path.join(base_dir, 'data/newmask'),
                                                      color_mode='grayscale',
                                                      batch_size=VAL_BATCH_SIZE,
                                                      target_size=IMG_SHAPE[:2],
                                                      subset='validation',
                                                      class_mode=None, seed=seed)

    train_generator = zip(train_img_generator, train_mask_generator)
    val_generator = zip(val_img_generator, val_mask_generator)
    return train_generator, val_generator


# model = FCN(input_shape=IMG_SHAPE)
model = Unet(input_shape=IMG_SHAPE)
# model = Deeplabv3()

model_history = None
for i in range(5):
    # Data Generation
    img_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)
    seed = np.random.randint(0, 100)
    t_gen, v_gen = get_gens(img_datagen, mask_datagen, seed)

    x,y = next(t_gen)


    # grey_3_channel = cv2.cvtColor(y[0], cv2.COLOR_GRAY2BGR)
    # plt.subplot(2,1,1)
    # plt.imshow(x[0])
    # plt.subplot(2,1,2)
    # plt.imshow(grey_3_channel)
    # plt.show()
    #
    # exit()

    # Model
    model_name = 'unet'
    model = Unet(input_shape=IMG_SHAPE)
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])
    model, history = train(model, t_gen, steps=100, epochs=25,
                 val_gen=v_gen, val_steps=10, fold=i, model_name=model_name)
    if model_history is None:
        model_history = history.history
    else:
        model_history = merge_h(model_history, history.history)

model.save_weights(os.path.join(base_dir, 'models/' + model_name + '.h5'))
with open(os.path.join('models/'+model_name+'_metrics.json', 'w')) as f:
    json.dump(model_history, f)
