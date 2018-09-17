import numpy as np
import os
import cv2
import pickle
import threading as thr
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, jaccard_similarity_score, accuracy_score, precision_score, recall_score

pix_to_class = lambda x: 1 if np.sum(x) > 1 else 0
dinamic_fold_dir = lambda x: '/home/gnardari/Documents/dd/greening_seeker/vision/data/folds/fold{}-{}/{}/001'.format(*x)

def read_hsv_and_flatten(f_path):
    img = cv2.cvtColor(cv2.imread(f_path), cv2.COLOR_BGR2HSV)
    return img.reshape(-1, 3)

def get_images_and_masks(f_dir, m_dir):
    x = []
    y = []
    for img_name in os.listdir(f_dir):
        img_path = os.path.join(f_dir, img_name)
        mask_path = os.path.join(m_dir, img_name)
        x.extend(read_hsv_and_flatten(img_path))
        y.extend(list(map(pix_to_class, read_hsv_and_flatten(mask_path))))
    return np.array(x), np.array(y)

# running static 3fold-cross-validation
# for i in range(1,4):
def train_fold(i):
    print('[{}] Initializing...'.format(i))
    t_frames_dir = dinamic_fold_dir((i, 'frames', 'train'))
    t_masks_dir = dinamic_fold_dir((i, 'masks', 'train'))
    x_train, y_train = get_images_and_masks(t_frames_dir, t_masks_dir)

    idx = np.random.permutation(int(0.1*len(y_train)))
    x_train = x_train[idx]
    y_train = y_train[idx]

    e_frames_dir = dinamic_fold_dir((i, 'frames', 'validation'))
    e_masks_dir = dinamic_fold_dir((i, 'masks', 'validation'))
    x_eval, y_eval = get_images_and_masks(e_frames_dir, e_masks_dir)

    print(np.shape(x_train))
    print(np.shape(y_train))

    print('[{}] Training...'.format(i))
    clf = SVC(kernel='poly', degree=4, n_iter=1000000)
    clf.fit(x_train,y_train)
    print('[{}] Evaluating...'.format(i))
    res = clf.predict(x_eval)

    acc = accuracy_score(y_eval, res)
    f1 = f1_score(y_eval, res)
    jac = jaccard_similarity_score(y_eval, res)
    prec = precision_score(y_eval, res)
    rec = recall_score(y_eval, res)

    scores = [acc, f1, jac]
    print('[{}] Acc: {}, f1: {}, jac: {}, prec: {}, rec: {}'.format(i, acc, f1, jac, prec, rec))

    with open('svm{}.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    return scores

with open('svm{}.pkl'.format(2), 'rb') as f:
    clf = pickle.load(f)

img = read_hsv_and_flatten('examples/image_1587.jpg')

mask = clf.predict(img)
mask[mask > 0.5] = 255
mask = mask.reshape(512,512)
cv2.imwrite('svmMask2.jpg', mask)

with open('svm{}.pkl'.format(3), 'rb') as f:
    clf = pickle.load(f)

mask = clf.predict(img)
mask[mask > 0.5] = 255
mask = mask.reshape(512,512)
cv2.imwrite('svmMask3.jpg', mask)
