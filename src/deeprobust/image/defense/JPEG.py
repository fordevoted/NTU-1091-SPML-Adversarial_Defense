import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import random


# from loaddata import load_mnist, load_cifar


def jpeg(X, image_size=32, window=8, channels=3, quality=75):
    X_jpeg = np.zeros(X.shape)
    # X = X.reshape((-1, image_size, image_size, channels))
    if channels == 1:
        mode = 'L'
        X = X.reshape((-1, image_size, image_size))
    elif channels == 3:
        mode = 'RGB'
    devide = window
    for i in range(len(X)):

        for h in range((int(image_size / devide))):
            for w in range((int(image_size / devide))):
                f = BytesIO()
                im = np.uint8(X[i, h * devide:(h + 1) * devide, w * devide:(w + 1) * devide] * 255)
                random.seed(0)
                Image.fromarray(im, mode=mode).save(f, "jpeg", quality=int(quality + int((random.random() - 0.5)*5) * 1))
                im_jpeg = Image.open(f)
                im_jpeg = np.array(im_jpeg.getdata())
                im_jpeg = im_jpeg.reshape((devide, devide, channels))
                X_jpeg[i, h * devide:(h + 1) * devide, w * devide:(w + 1) * devide] = im_jpeg / 255

                # cv2.imshow("split", cv2.resize(X[i, h*devide:(h+1)*devide, w*devide:(w+1)*devide],(256, 256)))
                # cv2.imshow("now", cv2.resize(X_jpeg[i, :],(256, 256)))
                # cv2.waitKey(0)
        # cv2.imshow("original", cv2.resize(X[i, :],(256, 256)))
        # cv2.imshow("after", cv2.resize(X_jpeg[i, : ], (256, 256)))
        # cv2.waitKey(0)
    return X_jpeg
