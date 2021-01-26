import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(".")

from utils.data import get_cifar10_loaders
from utils.vis import plot_loss, imsave, Logger
from utils.loss import FLPLoss, KLDLoss
from models.simple_vae import VAE

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/vae-123")
parser.add_argument('--batch_train', type=int, default=64)
parser.add_argument('--batch_test', type=int, default=16)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--initial_lr', type=float, default=0.0005)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--model', type=str, default="vae-123", choices=["vae-123", "vae-345", "pvae"])
args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataloader
dataloaders = get_cifar10_loaders(args.batch_train, args.batch_test)
# Model
model = VAE(device=device).to(device)

# Reconstruction loss
if args.model == "pvae":
    reconst_criterion = nn.MSELoss(reduction='sum')
elif args.model == "vae-123" or args.model == "vae-345":
    reconst_criterion = FLPLoss(args.model, device, reduction='sum')
# KLD loss
kld_criterion = KLDLoss(reduction='sum')

# Solver
optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
# Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
# Log
logdir = args.logdir
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Logger
logger = Logger(os.path.join(logdir, "log.txt"))
# History
history = {"train": [], "test": []}

# Save config
logger.write('----- Options ------')
for k, v in sorted(vars(args).items()):
    logger.write('%s: %s' % (str(k), str(v)))

# Start training
for epoch in range(args.epochs):
    for phase in ["train", "test"]:
        if phase == "train":
            model.train(True)
            logger.write(f"\n----- Epoch {epoch+1} -----")
        else:
            model.train(False)

        # Loss
        running_loss = 0.0
        # Data num
        data_num = 0

        # Train
        for i, data in enumerate(dataloaders[phase]):
            # Optimize params
            if phase == "train":
                optimizer.zero_grad()
                x = data[0]
                y = data[1]
                # Pass forward
                x = x.to(device)
                rec_x, mean, logvar = model(x)

                # Calc loss
                reconst_loss = reconst_criterion(x, rec_x)
                kld_loss = kld_criterion(mean, logvar)
                loss = args.alpha * kld_loss + args.beta * reconst_loss

                loss.backward()
                optimizer.step()

                # Visualize
                if i == 0 and x.size(0) >= 64:
                    imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", f"train.png"), 8, 8)
    
            elif phase == "test":
                with torch.no_grad():
                    optimizer.zero_grad()
                    x = data[0]
                    y = data[1]
                    # Pass forward
                    x = x.to(device)
                    rec_x, mean, logvar = model(x)

                    # Calc loss
                    reconst_loss = reconst_criterion(x, rec_x)
                    kld_loss = kld_criterion(mean, logvar)
                    loss = args.alpha * kld_loss + args.beta * reconst_loss

                    # Visualize
                    if x.size(0) >= 16:
                        imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", f"test-{i}.png"), 4, 4)

            # Add stats
            running_loss += loss # * x.size(0)
            data_num += x.size(0)

        # Log
        epoch_loss = running_loss / data_num
        logger.write(f"{phase} Loss : {epoch_loss:.4f}")
        history[phase].append(epoch_loss)

        if phase == "test":
            plot_loss(logdir, history)

# Save the model
torch.save(model.state_dict(),\
    os.path.join(logdir, 'final_model.pth'))