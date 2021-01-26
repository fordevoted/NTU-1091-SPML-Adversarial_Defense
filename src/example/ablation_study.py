import cv2
import numpy as np
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
root = os.path.dirname(parentdir)
sys.path.insert(0, root)
print(parentdir)
# sys.path.insert(0, '../src/deeprobust')

from src.deeprobust.image.defense.pgdtraining import PGDtraining
from src.deeprobust.image.attack.pgd import PGD
import torch
from torchvision import datasets, transforms
import src.deeprobust.image.netmodels.resnet as resnet
import src.deeprobust.image.defense.JPEG as JPEG
import os
from src.deeprobust.image.defense.VAE.models.simple_vae import VAE

"""
LOAD DATASETS
"""

# benign_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder('./examples/image/benign_examples', transform=transforms.Compose([transforms.ToTensor()])),
#     batch_size=10,
#     shuffle=True
# )
# adversary_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder('./examples/image/adversary_examples', transform=transforms.Compose([transforms.ToTensor()])),
#     batch_size=10,
#     shuffle=True
# )

benign_path = './benign_examples'
adversary_path = './pgd_adversary_examples'

benign_files = os.listdir(benign_path)
adversary_files = os.listdir(adversary_path)

data_files = adversary_files
data_path = adversary_path


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
"""
Test model 
"""
undefense_model = resnet.ResNet50().to('cuda')
print("Load network")

undefense_model.load_state_dict(torch.load(root+r"/model_weight/CIFAR10_ResNet50_epoch_190.pt"))
undefense_model.eval()

defense_model = resnet.ResNet50().to('cuda')
defense_model.load_state_dict(torch.load(root+r"/model_weight/CIFAR10_Resnet50_defense_epoch_140.pt"))
defense_model.eval()

transform_val = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10("data/cifar10",
                           train=False, download=True,
                           transform=transform_val)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


print('====== START EVAL BENIGN MODEL =====')
# w/o bilateral filtering
acc = 0
for f in data_files:
    x = cv2.imread(data_path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    # y[5:25, 5:25] = x[5:25, 5:25]
    # x = y
    # x = cv2.equalizeHist(x)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    # size = 2
    # kernel = np.ones((size, size), np.float32) / (size * size)
    # x = cv2.filter2D(x, -1, kernel)

    x = np.expand_dims(x, axis=0)
    x = JPEG.jpeg(x / 255, image_size=32, channels=3, quality=75)

    x *= 0.9
    x = np.power(x, 0.9)

    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()
    label = f.split('_')[-2]
    # label = classes[classify]

    predict0 = defense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    if classes[predict0[0][0]] == label:
        acc += 1
print("w/o bilateral filtering match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(data_files)) * 100))

# w/o transform to gray scale
acc = 0
for f in data_files:
    x = cv2.imread(data_path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)

    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.bilateralFilter(x, 20, 25, 25)

    x = np.expand_dims(x, axis=0)
    x = JPEG.jpeg(x / 255, image_size=32, channels=3, quality=75)

    x *= 0.9
    x = np.power(x, 0.9)

    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()
    label = f.split('_')[-2]
    # label = classes[classify]

    predict0 = defense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    if classes[predict0[0][0]] == label:
        acc += 1
print("w/o transform to gray scale defense match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(data_files)) * 100))

# w/o SLQ
acc = 0
for f in data_files:
    x = cv2.imread(data_path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)

    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    # y[5:25, 5:25] = x[5:25, 5:25]
    # x = y
    # x = cv2.equalizeHist(x)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    # size = 2
    # kernel = np.ones((size, size), np.float32) / (size * size)
    # x = cv2.filter2D(x, -1, kernel)

    x = np.expand_dims(x / 255, axis=0)

    x *= 0.9
    x = np.power(x, 0.9)

    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()
    label = f.split('_')[-2]
    # label = classes[classify]

    predict0 = defense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    if classes[predict0[0][0]] == label:
        acc += 1
print("w/o SLQ defense match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(data_files)) * 100))

# w/o gamma correction
acc = 0
for f in data_files:
    x = cv2.imread(data_path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)

    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    # y[5:25, 5:25] = x[5:25, 5:25]
    # x = y
    # x = cv2.equalizeHist(x)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    # size = 2
    # kernel = np.ones((size, size), np.float32) / (size * size)
    # x = cv2.filter2D(x, -1, kernel)

    x = np.expand_dims(x, axis=0)
    x = JPEG.jpeg(x / 255, image_size=32, channels=3, quality=75)

    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()
    label = f.split('_')[-2]
    # label = classes[classify]

    predict0 = defense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    if classes[predict0[0][0]] == label:
        acc += 1
print("w/o gamma correction defense match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(data_files)) * 100))

# w/o PGD adversarial training
acc = 0
for f in data_files:
    x = cv2.imread(data_path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)

    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    # y[5:25, 5:25] = x[5:25, 5:25]
    # x = y
    # x = cv2.equalizeHist(x)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    # size = 2
    # kernel = np.ones((size, size), np.float32) / (size * size)
    # x = cv2.filter2D(x, -1, kernel)

    x = np.expand_dims(x, axis=0)
    x = JPEG.jpeg(x / 255, image_size=32, channels=3, quality=75)

    x *= 0.9
    x = np.power(x, 0.9)

    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()
    label = f.split('_')[-2]
    # label = classes[classify]

    predict0 = undefense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    if classes[predict0[0][0]] == label:
        acc += 1
print("w/o PGD adversarial training defense match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(data_files)) * 100))

# Proposed defense
acc = 0
for f in data_files:
    x = cv2.imread(data_path + '/' + f)
    # y = np.ones((32, 32))*255
    # y = y.astype(np.uint8)

    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.bilateralFilter(x, 20, 25, 25)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    # y[5:25, 5:25] = x[5:25, 5:25]
    # x = y
    # x = cv2.equalizeHist(x)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

    # size = 2
    # kernel = np.ones((size, size), np.float32) / (size * size)
    # x = cv2.filter2D(x, -1, kernel)

    x = np.expand_dims(x, axis=0)
    x = JPEG.jpeg(x, image_size=32, channels=3, quality=75)



    x = x.swapaxes(2, 1).swapaxes(3, 1)
    x = torch.from_numpy(x).to('cuda').float()
    label = f.split('_')[-2]
    # label = classes[classify]

    predict0 = defense_model(x)
    predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
    if classes[predict0[0][0]] == label:
        acc += 1
print("Proposed defense match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(data_files)) * 100))

print('====== FINISH EVALUATEING =====')


# acc = 0
# for f in adversary_files:
#     x = cv2.imread(adversary_path + '/' + f)
#
#     # y = np.ones((32, 32)) * 255
#     # y = y.astype(np.uint8)
#
#     # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     # x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
#     # y[5:25, 5:25] = x[5:25, 5:25]
#     # x = y
#     # x = cv2.equalizeHist(x)
#     # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
#
#     x = np.expand_dims(x/255, axis=0).swapaxes(2, 1).swapaxes(3, 1)
#     x = torch.from_numpy(x).to('cuda').float()
#     label = f.split('_')[-2]
#     # label = classes[classify]
#
#     predict0 = defense_model(x)
#     predict0 = predict0.argmax(dim=1, keepdim=True).cpu().numpy()
#     if classes[predict0[0][0]] == label:
#         acc += 1
# print("Defense match: {match},  accuracy is: {acc}%".format(match=acc, acc=(acc / len(adversary_files)) * 100))
