import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import  os
import numpy as np
import matplotlib.pyplot as plt
import utils
train_on_gpu = torch.cuda.is_available()
def get_train_loader(image_type, image_dir='summer2winter_yosemite',
                     image_size = 128, batch_size = 16, num_workers = 0):
    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset,batch_size= batch_size,
                             shuffle=True, num_workers=num_workers)
    return train_loader, test_loader

G_XtoY, G_YtoX, D_X, D_Y = utils.create_model()


def train(n_epochs):
    dataloader_X, test_dataloader_X = get_train_loader(image_type='summer')
    dataloader_Y, test_dataloader_Y = get_train_loader(image_type='winter')
    # defining the optimizers
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    losses = []
    print_every = 10
    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())

    # create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(),lr,[beta1, beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1,beta2])

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs):
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = utils.scale(images_X)
        images_Y, _ = iter_Y.next()
        images_Y = utils.scale(images_Y)
        if train_on_gpu:
            images_X = images_X.cuda()
            images_Y = images_Y.cuda()

        # discriminator training

        d_x_optimizer.zero_grad()
        out_x = D_X(images_X)
        d_x_real_loss = utils.real_mse_loss(out_x)

        fake_x = G_YtoX(images_Y)
        out_x_fake = D_X(fake_x)
        d_x_fake_loss = utils.fake_mse_loss(out_x_fake)

        d_x_loss = d_x_real_loss + d_x_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()

        d_y_optimizer.zero_grad()
        out_y = D_Y(images_Y)
        d_y_real_loss = utils.real_mse_loss(out_y)

        fake_y = G_XtoY(images_X)
        out_y_fake = D_Y(fake_y)
        d_y_fake_loss = utils.fake_mse_loss(out_y_fake)
        d_y_loss = d_y_real_loss + d_y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()

        # generator training
        g_optimizer.zero_grad()

        fake_x = G_YtoX(images_X)
        out_x_fake = D_X(fake_x)
        g_x_loss = utils.real_mse_loss(out_x_fake)

        fake_y = G_XtoY(images_Y)
        out_y_fake = D_Y(fake_y)
        g_y_loss = utils.real_mse_loss(out_y_fake)

        reconstructed_y = G_XtoY(fake_y)
        reconstructed_y_loss = utils.cycle_consistency_loss(images_Y, reconstructed_y, lambda_weight = 10)
        reconstructed_x = G_YtoX(fake_x)
        reconstructed_x_loss = utils.cycle_consistency_loss(images_X,reconstructed_x, lambda_weight=10)
        g_total_loss = g_x_loss + g_y_loss + reconstructed_x_loss + reconstructed_y_loss
        g_total_loss.backward()
        g_optimizer.step()

        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
        # saving the models
    PATH_G_X_to_Y = './G_X_to_Y.pth'
    torch.save(G_XtoY.state_dict(), PATH_G_X_to_Y)
    PATH_G_Y_to_X = './G_Y_to_X.pth'
    torch.save(G_YtoX.state_dict(), PATH_G_Y_to_X)
        
    return losses

if __name__ == "__main__":
    n_epochs = 4000
    losses = train(n_epochs)
