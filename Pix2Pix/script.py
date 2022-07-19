import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from masks import BasicDataset
import random
from tqdm import tqdm
import numpy as np
from model import StarGAN

global_step = 0
PATH = 'checkpoints/'
name = 'GAN1'

def generate_example(model, loader):
    model.eval()

    rand_index = random.randrange(len(loader.dataset))
    image, mask = loader.dataset[rand_index]
    
    device = model.device
    image  = image.to(device) 
    
    
    with torch.no_grad():
        reconstructed_image = model.generate(image.unsqueeze(0)).squeeze(0)

    
    image = ((image.permute(1, 2, 0) + 1) / 2).cpu().numpy()
    mask = ((mask.permute(1, 2, 0) + 1) / 2).cpu().numpy()
    reconstructed_image = ((reconstructed_image.permute(1, 2, 0) + 1) / 2).cpu().numpy()

    
    return image, reconstructed_image, mask


def train(epoch, loader, model, n_disc=1):
    global global_step
    d_losses = []
    g_losses = []
    model.train()
    device = model.device
    for i, (image, mask) in tqdm(enumerate(loader), desc=f"trainloop: {epoch}", leave=False):
        image, mask = image.to(device), mask.to(device)

        d_loss = model.trainD(image, mask)
        g_loss = model.trainG(image, mask)
        
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if (global_step + 1) % 150 == 0:
            # example = generate_example(model, loader)
            g_train_loss = np.mean(g_losses)
            d_train_loss = np.mean(d_losses)
            d_losses = []
            g_losses = []
                    
            # example.update({'G loss': g_train_loss,
            #                 'D loss': d_train_loss
            #                 })

            # wandb.log(example)
            print('G_loss:', g_train_loss, 'D_loss:', d_train_loss)
            model.train()
            torch.save(model.G.state_dict(), PATH + name + "_generator.pt")
            torch.save(model.D.state_dict(), PATH + name + "_critic.pt")

        global_step += 1

dir_img = 'other_data/crop_imgs/'
dir_mask = 'other_data/crop_masks/'

batch_size = 2
val_percent = 0.1

dataset = BasicDataset(dir_img, dir_mask, size=256)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val

trainset, valset = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False)

device = torch.device('cuda:1')
model = StarGAN(device, d_n_layers=2, g_n_layers=2, c_dim=1, image_size=256)

epochs = 400
for i in range(1, epochs + 1):
    train(i, train_loader, model)

for i in range(3):
    im, gen, mask = generate_example(model, train_loader)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im)
    ax[1].imshow(gen)
    ax[2].imshow(mask)
    plt.savefig('sample_' + str(i))
