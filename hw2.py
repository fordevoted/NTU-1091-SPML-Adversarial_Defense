import os
import sys
import random
import src.deeprobust.image.netmodels.resnet as resnet
import torch
import cv2
import numpy as np
import src.deeprobust.image.defense.JPEG as JPEG

sys.path.insert(0, './src')
if len(sys.argv) >= 2:
    path = sys.argv[1]
else:
    path = './example_folder'
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

files = os.listdir(path)
fp = open('predict.txt', 'w+')

defense_model = resnet.ResNet50().to('cuda')
defense_model.load_state_dict(torch.load("./model_weight/CIFAR10_Resnet50_defense_epoch_140.pt"))
defense_model.eval()

for f in files:
    x = cv2.imread(path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)
    if x.max() <= 1 :
        x = (x * 255).astype(np.uint8)
    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

    x = np.expand_dims(x, axis=0)
    x = JPEG.jpeg(x / 255, image_size=32, channels=3, quality=75)

    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()

    predict0 = defense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    print(classes[predict0[0][0]], file=fp)
