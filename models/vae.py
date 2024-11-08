import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from .encoders import *
from .PreResNet import *



class BaseVAE(nn.Module):
    def __init__(self, feature_dim=28, num_hidden_layers=1, hidden_size=32, z_dim =4, dim_times = 8, num_classes=10):
        super().__init__()
        self.z_dim = z_dim
        self.dim_times = dim_times
        self.z_encoder = Z_Encoder(feature_dim=feature_dim, num_classes=num_classes, num_hidden_layers=num_hidden_layers, hidden_size = hidden_size, z_dim=z_dim*self.dim_times, s_dim=z_dim*self.dim_times, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = X_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size, z_dim=2*z_dim*self.dim_times)
        self.num_classes = num_classes
        self.lambda_dim = z_dim*(z_dim-1)//2
        self.beta_m = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.beta_logvar = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.z_weight = nn.Sequential(nn.Linear(num_classes, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.lambda_dim, bias=False))
        self.t_decoder = nn.Sequential(nn.Linear(z_dim*self.dim_times,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, num_classes))
        self.register_parameter(name='M_x', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
        self.register_parameter(name='M_y', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))


    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits, dim=1)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
    
    def _get_mean_logvar(self, beta1, beta2, weight):
        z_list=[]
        mean_list = []
        for i in range(self.z_dim):
            mean=beta1[:,i]
            logvar=beta2[:,i]
            for j in range(i):
                begin_index = i*(i-1)//2
                mean+=weight[:,begin_index+j]*z_list[j]
            mean_list.append(mean)
            z = self._z_reparameterize(mean, logvar)
            z_list.append(z)
        
        mean_list=torch.stack(mean_list).T
        z_list=torch.stack(z_list).T

        return z_list, mean_list, beta2
    
    def _get_mean_logvar_mul(self, beta1, beta2, weight):
        z_list=[]
        mean_list = []
        for i in range(self.z_dim):
            mean=beta1[:,i]
            logvar=beta2[:,i]
            for j in range(i):
                begin_index = i*(i-1)//2
                mean+=weight[:,begin_index+j]*z_list[j]
            mean_list.append(mean)
            z = self._z_reparameterize(mean, logvar)
            z_list.append(z)
        
        mean_list=torch.stack(mean_list, dim=1)
        z_list=torch.stack(z_list, dim=1)

        return z_list, mean_list, beta2

    def forward(self, x: torch.Tensor, net): 
        bz = x.shape[0]
        c_logits = net(x)
        y_hat = self._y_hat_reparameterize(c_logits)
        p_beta1 = self.beta_m(y_hat)
        p_beta2 = self.beta_logvar(y_hat)
        p_weight = self.z_weight(y_hat)
        beta1, beta2, weight, mu_s, log_var_s = self.z_encoder(x, y_hat) # bz * self.dim_times*self.dim_times
        beta1 = beta1.view(bz, self.z_dim, self.dim_times)
        beta2 = beta2.view(bz, self.z_dim, self.dim_times)
        _, p_z_m, p_z_v = self._get_mean_logvar(p_beta1, p_beta2, p_weight) # bz * self.dim_times
        p_z_m = p_z_m.unsqueeze(2)
        p_z_v = p_z_v.unsqueeze(2)
        weight = weight.unsqueeze(2)
        z, z_mean, z_logvar = self._get_mean_logvar_mul(beta1, beta2, weight)
        in_z_x = torch.mul(z, self.M_x.view(1, self.z_dim, 1))
        in_z_x = in_z_x.view(bz, self.z_dim*self.dim_times)
        s = self._z_reparameterize(mu_s, log_var_s)
        x = self.x_decoder(torch.concat([in_z_x, s], dim=1))
        in_z_y = torch.mul(z, self.M_y.view(1, self.z_dim, 1))
        in_z_y = in_z_y.view(bz, self.z_dim*self.dim_times)
        n_logits = self.t_decoder(in_z_y)

        return x, n_logits, c_logits, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v, self.M_x, self.M_y


class VAE_FASHIONMNIST(BaseVAE):
    def __init__(self, feature_dim=28, input_channel=1, z_dim =4, dim_times = 8, num_classes=10):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dim_times = dim_times
        self.z_encoder = CONV_Encoder_FMNIST(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim*self.dim_times, s_dim=z_dim*self.dim_times, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = CONV_Decoder_FMNIST(num_classes=num_classes, z_dim=2*z_dim*self.dim_times)
        self.lambda_dim = z_dim*(z_dim-1)//2
        self.beta_m = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.beta_logvar = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.z_weight = nn.Sequential(nn.Linear(num_classes, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.lambda_dim, bias=False))
        self.t_decoder = nn.Sequential(nn.Linear(z_dim*self.dim_times,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, num_classes))
        self.register_parameter(name='M_x', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
        self.register_parameter(name='M_y', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))


class VAE_CIFAR100(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim = 4, dim_times = 8, num_classes=100):
        super().__init__()
        self.z_dim = z_dim
        self.dim_times = dim_times
        self.num_classes = num_classes
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim*self.dim_times, s_dim=z_dim*self.dim_times, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=2*z_dim*self.dim_times)
        self.lambda_dim = z_dim*(z_dim-1)//2
        self.beta_m = nn.Sequential(nn.Linear(num_classes,256),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256,256),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256, self.z_dim))
        self.beta_logvar = nn.Sequential(nn.Linear(num_classes,256),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256,256),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256, self.z_dim))
        self.z_weight = nn.Sequential(nn.Linear(num_classes, 256, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256, 256, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256, self.lambda_dim, bias=False))
        self.t_decoder = nn.Sequential(nn.Linear(z_dim*self.dim_times,256),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256,256),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(256, num_classes))
        self.register_parameter(name='M_x', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
        self.register_parameter(name='M_y', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))

class VAE_CIFAR10(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim = 4, dim_times = 8, num_classes=10):
        super().__init__()
        self.z_dim = z_dim
        self.dim_times = dim_times
        self.num_classes = num_classes
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim*self.dim_times, s_dim=z_dim*self.dim_times, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=2*z_dim*self.dim_times)
        self.lambda_dim = z_dim*(z_dim-1)//2
        self.beta_m = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.beta_logvar = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.z_weight = nn.Sequential(nn.Linear(num_classes, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.lambda_dim, bias=False))
        self.t_decoder = nn.Sequential(nn.Linear(z_dim*self.dim_times,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, num_classes))
        self.register_parameter(name='M_x', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
        self.register_parameter(name='M_y', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
    

class VAE_CLOTHING1M(BaseVAE):
    def __init__(self, feature_dim=224, input_channel=3, z_dim =4, dim_times=64, num_classes=14):
        super().__init__()
        self.z_dim = z_dim
        self.dim_times = dim_times
        self.z_encoder = CONV_Encoder_CLOTH1M(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim*self.dim_times, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = CONV_Decoder_CLOTH1M(num_classes=num_classes, z_dim=z_dim*self.dim_times)
        self.num_classes = num_classes
        self.lambda_dim = z_dim*(z_dim-1)//2
        self.beta_m = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.beta_logvar = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.z_weight = nn.Sequential(nn.Linear(num_classes, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, 64, bias=False),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.lambda_dim, bias=False))
        self.t_decoder = nn.Sequential(nn.Linear(z_dim*self.dim_times,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, num_classes))
        self.register_parameter(name='M_x', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
        self.register_parameter(name='M_y', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))


def VAE_SVHN(feature_dim=32, input_channel=3, z_dim = 4, num_classes=10):
    return VAE_CIFAR10(feature_dim=feature_dim, input_channel=input_channel, z_dim =z_dim, num_classes=num_classes)


def VAE_MNIST( feature_dim=28, input_channel=1, z_dim = 4, num_classes=10):
    return VAE_FASHIONMNIST(feature_dim=feature_dim, input_channel=input_channel, z_dim =z_dim, num_classes=num_classes)


class VAE_WEBVISION(BaseVAE):
    def __init__(self, feature_dim=299, num_hidden_layers=1, hidden_size=25, z_dim =4, dim_times=64, num_classes=50, input_channel = 3):
        super().__init__()
        self.z_dim = z_dim
        self.dim_times = dim_times
        self.z_encoder = CONV_Encoder_WEBVISION(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim*self.dim_times, s_dim=z_dim*self.dim_times, lambda_dim = z_dim*(z_dim-1)//2)
        self.x_decoder = CONV_Decoder_WEBVISION(num_classes=num_classes, z_dim=2*z_dim*self.dim_times)
        self.num_classes = num_classes
        self.lambda_dim = z_dim*(z_dim-1)//2
        self.beta_m = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.beta_logvar = nn.Sequential(nn.Linear(num_classes,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.z_weight = nn.Sequential(nn.Linear(num_classes, 64, bias=False),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, 64, bias=False),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.lambda_dim, bias=False))
        self.t_decoder = nn.Sequential(nn.Linear(z_dim*self.dim_times,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, num_classes))
        self.register_parameter(name='M_x', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
        self.register_parameter(name='M_y', param=nn.parameter.Parameter(torch.ones(z_dim).cuda()))
