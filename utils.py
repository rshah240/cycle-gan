import torch
import torch.nn as nn
import torch.nn.functional as F

def scale(x,feature_range= (-1,1)):
    min,max = feature_range
    x = x*(max - min) + min
    return x

def real_mse_loss(D_out):
    '''
    :param D_out: Discriminator Output
    :return: loss function
    '''
    return torch.mean((D_out - 1)**2)

def fake_mse_loss(D_out):
    '''
    :param D_out: Discriminator Output
    :return: loss function
    '''
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    '''Reconstruction Loss Function'''
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight*reconstr_loss


def conv(in_channels, out_channels, kernel_size = 4, stride=2, padding=1,
         batch_norm = True):
    '''

    :param in_channels: input channels
    :param out_channels: output channels
    :param kernel_size: window size
    :param stride: stride number
    :param padding: padding
    :param batch_norm: batch_norm boolean value
    :return: convolution layers
    '''

    layers = []
    conv_layers = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride = stride, padding = padding)

    layers.append(conv_layers)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    '''Discriminator Architecture'''
    def __init__(self, conv_dim = 64):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, batch_norm=False)
        self.conv2 = conv(in_channels= conv_dim, out_channels=conv_dim*2)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4)
        self.conv4 = conv(in_channels=conv_dim*4,out_channels=conv_dim*8)

        self.conv5 = conv(in_channels=conv_dim*8, out_channels=1)

    def forward(self, x):
        x = x.view(-1,3,128,128)

        out = F.leaky_relu(self.conv1(x),0.2)
        out = F.leaky_relu(self.conv2(out),0.2)
        out = F.leaky_relu(self.conv3(out),0.2)
        out = F.leaky_relu(self.conv4(out),0.2)
        out = self.conv5(out)

        return out

class ResidualBlock(nn.Module):
    '''Defines a Residual Block Architecture'''
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(in_channels=conv_dim, out_channels=conv_dim,
                          kernel_size=3, stride=1, padding=1, batch_norm=True)

        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3,
                          stride = 1, padding=1, batch_norm=True)

    def forward(self, x):

        out_1 = F.relu(self.conv1(x))
        out_2 = x + self.conv2(out_1)

        return out_2

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm =True):
    '''Creates a transpose convolution layer, with optional batch normalization option'''
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class CycleGenerator(nn.Module):
    '''Defines Generator Block Architecture'''
    def __init__(self, conv_dim = 64, n_res_blocks = 6):
        super(CycleGenerator,self).__init__()

        self.conv1 = conv(3,conv_dim,4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        
        res_layers = []
        for i in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))

        self.res_block = nn.Sequential(*res_layers)
        
        self.deconv1 = deconv(in_channels=conv_dim*4, out_channels=conv_dim*2, kernel_size=4)
        self.deconv2 = deconv(in_channels=conv_dim*2,out_channels=conv_dim,kernel_size=4)
        self.deconv3 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4)

    def forward(self, x):
        x = x.view(-1, 3, 128,128)
        out = F.leaky_relu(self.conv1(x),0.2)
        out = F.leaky_relu(self.conv2(out),0.2)
        out = F.leaky_relu(self.conv3(out),0.2)

        out = self.res_block(out)

        out = F.leaky_relu(self.deconv1(out), 0.2)
        out = F.leaky_relu(self.deconv2(out), 0.2)
        out = torch.tanh(self.deconv3(out))

        return  out


def create_model(g_conv_dim = 64, d_conv_dim = 64, n_res_blocks = 6):
    '''Builds the generators and discriminators'''

    # Instantiate Generators
    G_XtoY = CycleGenerator(conv_dim = g_conv_dim,n_res_blocks = n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim = g_conv_dim,n_res_blocks = n_res_blocks)

    # Instantiate discriminators
    D_X = Discriminator(conv_dim=d_conv_dim)
    D_Y = Discriminator(conv_dim=d_conv_dim)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        device = torch.device('cuda:0')
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU')
    else:
        print('Train on CPU')

    return G_XtoY, G_YtoX, D_X, D_Y









        





