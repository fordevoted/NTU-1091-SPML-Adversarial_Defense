import inspect
import os
import random
import sys

import cv2
import numpy as np
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
root = os.path.dirname(parentdir)
sys.path.insert(0, root)
print(parentdir)
from torchvision import datasets, transforms
from src.deeprobust.image.attack.pgd import PGD
import src.deeprobust.image.netmodels.resnet as resnet
from src.deeprobust.image.config import attack_params
import os

os.environ["cuda_visible_devices"] = '0'
# for i in range(0, 10):
#     os.mkdir(r"C:\Users\USER\Desktop\Security of ML\hw2\Adversarial-training\src\examples\image\training_adversary_examples" + '\\'+str(i))
model = resnet.ResNet50().to('cuda')
print("Load network")

model.load_state_dict(torch.load(root+r"/model_weight/CIFAR10_ResNet50_epoch_190.pt"))
model.eval()

transform_val = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10("data/cifar10",
                           train=False, download=True,
                           transform=transform_val)
batch_size = 1
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # , **kwargs)

classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))
iteration = int(len(dataset) / batch_size)
counter = 0
print(iteration)
for i in range(iteration):
    xx, yy = next(iter(test_loader))
    xx = xx.to('cuda').float()

    predict0 = model(xx)
    predict0 = predict0.argmax(dim=1, keepdim=True)

    adversary = PGD(model)
    AdvExArray = adversary.generate(xx, yy, **attack_params['PGD_CIFAR10']).float()

    predict1 = model(AdvExArray)
    predict1 = predict1.argmax(dim=1, keepdim=True)

    print('====== RESULT =====', i)
    # print('true label', classes[yy.cpu()])
    # print('predict_orig', classes[predict0.cpu()])
    # print('predict_adv', classes[predict1.cpu()])

    xx_save = xx.cpu().numpy().swapaxes(1, 3).swapaxes(1, 2)
    # print('xx:', x_show)
    # plt.imshow(x_show, vmin=0, vmax=255)
    # plt.savefig('./adversary_examples/cifar_advexample_orig.png')
    # print('x_show', x_show)

    # print('---------------------')
    AdvExArray = AdvExArray.cpu().detach().numpy()
    AdvExArray = AdvExArray.swapaxes(1, 3).swapaxes(1, 2)
    label_array = yy.cpu().numpy()
    for size in range(len(AdvExArray)):
        # print('Adv', AdvExArray)
        # print('----------------------')
        # print(AdvExArray)

        # if random.random() < 0.02:
        #     cv2.imwrite('./benign_examples/{counter}_{name}_{label}.png'.format(name=classes[int(label_array[size])], counter=counter, label=label_array[size]), xx_save[size] * 255)
        if random.random() < 0.1:

            cv2.imwrite('./fgsm_adversary_examples/{counter}_{name}_{label}.png'.format(name=classes[int(label_array[size])],
                                                                               counter=counter, label=label_array[size]),
                    AdvExArray[size] * 255)
            # cv2.imwrite('./pgd_adversary_examples/{counter}_{name}_{label}.png'.format(name=classes[int(label_array[size])],
            #                                                                    counter=counter, label=label_array[size]),
            #         xx_save[size] * 255)
        # else:
        #     cv2.imwrite('./testing_adversary_examples/{label}/{counter}_{name}_adv.png'.format(
        #         name=classes[int(label_array[size])],
        #         counter=counter, label=label_array[size]),
        #                 AdvExArray[size] * 255)
        #     cv2.imwrite('./testing_adversary_examples/{label}/{counter}_{name}.png'.format(
        #         name=classes[int(label_array[size])],
        #         counter=counter, label=label_array[size]),
        #         xx_save[size] * 255)
        counter += 1
