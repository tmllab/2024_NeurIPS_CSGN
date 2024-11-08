

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def make_hidden_layers(num_hidden_layers=1, hidden_size=5, prefix="y"):
    block = nn.Sequential()
    for i in range(num_hidden_layers):
        block.add_module(prefix+"_"+str(i), nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.BatchNorm1d(hidden_size),nn.ReLU()))
    return block



class CONV_Encoder_FMNIST(nn.Module):
    def __init__(self, in_channels =1, feature_dim = 28, num_classes = 2,  hidden_dims = [32, 64, 128, 256, 512, 512], z_dim = 2, s_dim = 2, lambda_dim=1):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_lambda = nn.Linear(hidden_dims[-1], lambda_dim, bias=False)
        self.fc_mu_s = nn.Linear(hidden_dims[-1], s_dim)
        self.fc_logvar_s = nn.Linear(hidden_dims[-1], s_dim)

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),1,self.feature_dim ,self.feature_dim )
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        weight = self.fc_lambda(x)
        mu_s = self.fc_mu_s(x)
        log_var_s = self.fc_logvar_s(x)
        return mu, log_var, weight, mu_s, log_var_s


class CONV_Decoder_FMNIST(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [512, 512, 256, 128, 64,32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0])
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               ),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 5))


    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 512, 1, 1)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out



class CONV_Encoder_CIFAR(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 32, num_classes = 2,  hidden_dims = [32, 64, 128, 256, 512, 512], z_dim = 2, s_dim = 2, lambda_dim=1):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_lambda = nn.Linear(hidden_dims[-1], lambda_dim, bias=False)
        self.fc_mu_s = nn.Linear(hidden_dims[-1], s_dim)
        self.fc_logvar_s = nn.Linear(hidden_dims[-1], s_dim)

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        weight = self.fc_lambda(x)
        mu_s = self.fc_mu_s(x)
        log_var_s = self.fc_logvar_s(x)
        return mu, log_var, weight, mu_s, log_var_s


class CONV_Decoder_CIFAR(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [512, 512, 256, 128, 64,32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0])
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], out_channels= hidden_dims[-1],
                                      kernel_size= 3, stride=1, padding= 1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1))


    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 512, 1, 1)
        out = self.decoder(out)
        out = self.final_layer(out)

        return out


class CONV_Encoder_CLOTH1M(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 224, num_classes = 2,  hidden_dims = [32, 64, 128, 256, 512, 512], z_dim = 2, lambda_dim=1):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims[:5]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 7, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        for h_dim in hidden_dims[5:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*16, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*16, z_dim)
        self.fc_lambda = nn.Linear(hidden_dims[-1]*16, lambda_dim, bias=False)

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)

        x = self.encoder(x)
    
        x = torch.flatten(x, start_dim=1)
       
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        weight = self.fc_lambda(x)
        return mu, log_var, weight


class CONV_Decoder_CLOTH1M(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [512, 512, 256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 16)
        modules = []

        for i in range(1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        for i in range(1, len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=9,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1))


    def forward(self, z):
        out = self.decoder_input(z)   
        out = out.view(-1, 512, 4, 4)
        out = self.decoder(out)  
        out = self.final_layer(out)
        return out


class CONV_Encoder_WEBVISION(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 299, num_classes = 2,  hidden_dims = [32, 64, 128, 256, 512], z_dim = 2, s_dim = 2, lambda_dim=1):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 7, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(nn.Linear(hidden_dims[-1]*36,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(hidden_dims[-1]*36,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, self.z_dim))
        self.fc_lambda = nn.Sequential(nn.Linear(hidden_dims[-1]*36,64, bias=False),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, lambda_dim, bias=False))
        self.fc_mu_s = nn.Sequential(nn.Linear(hidden_dims[-1]*36,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, s_dim))
        self.fc_logvar_s = nn.Sequential(nn.Linear(hidden_dims[-1]*36,64),
                                                                    nn.BatchNorm1d(64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, s_dim))

    def forward(self, x, y_hat):
        embedded_class = self.embed_class(y_hat)
        x = x.view(x.size(0),self.in_channels,self.feature_dim,self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = self.encoder(x)
     
        x = torch.flatten(x, start_dim=1)
       
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        weight = self.fc_lambda(x)
        mu_s = self.fc_mu_s(x)
        log_var_s = self.fc_logvar_s(x)
        return mu, log_var, weight, mu_s, log_var_s


class CONV_Decoder_WEBVISION(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [512, 256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 36)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=9, #9
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=12, #9
                                               stride=1,
                                               padding=0,
                                               output_padding=0),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1),#311
                            nn.Identity()
                            )


    def forward(self, z):
        out = self.decoder_input(z) 
        out = out.view(-1, 512, 6, 6)
        out = self.decoder(out) 
        out = self.final_layer(out)
        return out



class Z_Encoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5, z_dim = 2, s_dim = 2, lambda_dim=1):
        super().__init__()
        self.z_fc1 = nn.Linear(feature_dim+num_classes, hidden_size)
        self.z_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="z")
        self.z_fc_mu = nn.Linear(hidden_size, z_dim)  # fc21 for mean of Z
        self.z_fc_logvar = nn.Linear(hidden_size, z_dim)  # fc22 for log variance of Z
        self.fc_lambda = nn.Linear(hidden_size, lambda_dim, bias=False)
        self.fc_mu_s = nn.Linear(hidden_size, s_dim)
        self.fc_logvar_s = nn.Linear(hidden_size, s_dim)

    def forward(self, x, y_hat):
        out = torch.cat((x, y_hat), dim=1)
        out = F.relu(self.z_fc1(out))
        out = self.z_h_layers(out)
        mu = F.elu(self.z_fc_mu(out))
        logvar = F.elu(self.z_fc_logvar(out))
        weight = F.elu(self.fc_lambda(x))
        mu_s = self.fc_mu_s(x)
        log_var_s = self.fc_logvar_s(x)
        
        return mu, logvar, weight, mu_s, log_var_s


class X_Decoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, num_hidden_layers=1, hidden_size = 5, z_dim = 1):
        super().__init__()
        self.recon_fc1 = nn.Linear(z_dim, hidden_size)
        self.recon_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="recon")
        self.recon_fc2 = nn.Linear(hidden_size, feature_dim)    

    def forward(self, z):
        out = F.relu(self.recon_fc1(z))
        out = self.recon_h_layers(out)
        x = self.recon_fc2(out)
        return x