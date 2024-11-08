## Reference:
## 1. DivideMix: https://github.com/LiJunnan1992/DivideMix
## 2. CausalNL: https://github.com/a5507203/IDLN
## Our code is heavily based on the above-mentioned repositories. 

# Loading libraries
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import random 
import argparse
import numpy as np
from models.PreResNet import *
from models.vae import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import argparse
import os
import numpy as np
from tqdm import tqdm

# Default values
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--vae_lr', '--vae_learning_rate', default=0.001, type=float, help='initial vae learning rate')
parser.add_argument('--noise_mode',  default='instance')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--lambda_elbo', default=0.5, type=float, help='weight for elbo')
parser.add_argument('--lambda_M', default=0.001, type=float, help='weight for M')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--z_dim', default=4, type=int)
args,_ = parser.parse_known_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch,net,net2,optimizer,vae_model, vae_model2,optimizer_vae,labeled_trainloader,unlabeled_trainloader, train_all_loader, all_dataset, net_1 = True):
    net.train()
    net2.eval() #fix one network and train the other    
    vae_model.train()
    vae_model2.eval()
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    all_train_iter = iter(train_all_loader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, noisy_labels, w_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, noisy_labels, w_u = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x_onehot = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        noisy_labels = torch.zeros(inputs_u.size(0), args.num_class).scatter_(1, noisy_labels.view(-1,1), 1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)
        w_u = w_u.view(-1,1).type(torch.FloatTensor)

        labels_x = labels_x.cuda()
        inputs_x, inputs_x2, labels_x_onehot, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x_onehot.cuda(), w_x.cuda()
        inputs_u, inputs_u2, noisy_labels, w_u = inputs_u.cuda(), inputs_u2.cuda(), noisy_labels.cuda(), w_u.cuda()


        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)    
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)         
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x_onehot + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                    
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
        
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()      
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss_dm = Lx + lamb * Lu + penalty
        try:
            vae_data, clean_label, vae_targets, index = all_train_iter.next()
        except:
            all_train_iter = iter(train_all_loader)
            vae_data, clean_label, vae_targets, index = all_train_iter.next()
        if args.noise_mode != 'instance' and args.noise_mode != 'dependent':
            gt_t = all_dataset.t
            first_dim_indices = np.arange(vae_data.shape[0])
            gt_t = np.repeat(gt_t[np.newaxis, :, :], vae_data.shape[0], axis=0)
            gt_t = gt_t[first_dim_indices,clean_label,:]
        else:
            gt_t = all_dataset.t[index]
        
        # vae_targets = torch.zeros(vae_targets.size(0), args.num_class).scatter_(1, vae_targets.view(-1,1), 1)
        vae_loss, recons_loss, nce_loss, kld_loss, L1 = train_vae(vae_data, vae_targets, gt_t, net, vae_model, epoch, batch_idx, net_1)

        loss = loss_dm + args.lambda_elbo * vae_loss + args.lambda_M * L1

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f  Recons loss: %.2f  Noisy CE loss: %.2f  KL loss: %.2f  L1 loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item(), recons_loss.item(), nce_loss.item(), kld_loss.item(), L1.item()))
        sys.stdout.flush()
    return loss


# Train vae
def train_vae(vae_data, vae_targets, gt_t, net, vae_model, epoch, batch_idx, net_1):
    vae_model.train()
    data = vae_data.cuda()
    targets = vae_targets.cuda()
    #forward
    x, n_logits, c_logits, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v, M_x, M_y  = vae_model(data, net)
    # calculate L1
    L1 = torch.norm(M_x, p=1) + torch.norm(M_y, p=1)
    # calculate loss
    vae_loss, recons_loss, nce_loss, kld_loss = my_vae_loss(x, n_logits, data, targets, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v)

    return vae_loss, recons_loss, nce_loss, kld_loss, L1

# two component GMM model
def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))    
    eval_correct = []
    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]    
            pred = torch.max(outputs, 1)[1]
            eval_correct += (pred == targets).tolist()   
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss,eval_correct

# two component GMM model
def matrix_error(model):    
    model.eval()
    all_target = []
    all_preds = []
    eval_num = 0
    eval_acc  = 0
    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(eval_loader):
            eval_num += inputs.shape[0]
            all_target.append(targets)
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            pred = torch.max(outputs, 1)[1]
            all_preds.append(pred.cpu())
            eval_acc += (pred == targets).sum().item()   
    all_target = torch.cat(all_target)
    all_preds = torch.cat(all_preds)
    print(all_target.shape)
    print(all_target.shape)  
    print(eval_acc)

# Testing
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)   
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()        
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

# %%
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

# %%
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

# %%
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

# %%
def create_model():
    if args.dataset == 'fashionmnist' or args.dataset == 'mnist':
        model = ResNet18(num_classes=args.num_class, in_c=1)
    else:
        model = ResNet18(num_classes=args.num_class, in_c=3)
    
    if args.dataset=='cifar10' or args.dataset=='svhn':
        vae_model = VAE_CIFAR10(z_dim=args.z_dim, num_classes=10)
    elif args.dataset=='fashionmnist' or args.dataset == 'mnist':
        vae_model = VAE_FASHIONMNIST(z_dim=args.z_dim, num_classes=10)
    elif args.dataset=='cifar100':
        vae_model = VAE_CIFAR100(z_dim=args.z_dim, num_classes=100)
    
    total_params1 = sum(p.numel() for p in model.parameters())
    total_params2 = sum(p.numel() for p in vae_model.parameters())
    print(f"Number of parameters: {total_params1+total_params2}")
    model = model.cuda()
    vae_model = vae_model.cuda()
    return model, vae_model

# %%
os.makedirs('./checkpoint', exist_ok = True)

if not os.path.exists('./saved/'+args.dataset):
    os.system('mkdir -p %s'%('./saved/'+args.dataset))

# %%
if args.dataset == 'fashionmnist':
    IMG_MEAN = [0.1307, ]
    IMG_STD = [0.3081, ]
    warm_up = 10
    decay_epoch = 100
    args.num_class = 10
    args.data_path = './datasets/fashionmnist'
elif args.dataset == 'mnist':
    IMG_MEAN = [0.1307, ]
    IMG_STD = [0.3081, ]
    warm_up = 5
    decay_epoch = 10
    args.num_class = 10
    args.num_epochs = 20
    args.data_path = './datasets/mnist'
elif args.dataset == 'svhn':
    IMG_MEAN = [0.4914, 0.4822, 0.4465]
    IMG_STD = [0.2023, 0.1994, 0.2010]
    warm_up = 10
    decay_epoch = 30
    args.num_class = 10
    args.data_path = './datasets/svhn'
elif args.dataset == 'cifar10':
    IMG_MEAN = [0.4914, 0.4822, 0.4465]
    IMG_STD = [0.2023, 0.1994, 0.2010]
    warm_up = 10
    decay_epoch = 100
    args.num_class = 10
    args.data_path = './cifar-10'
elif args.dataset == 'cifar100':
    IMG_MEAN = [0.507, 0.487, 0.441]
    IMG_STD = [0.267, 0.256, 0.276]
    warm_up = 10
    decay_epoch = 100
    args.num_class = 100
    args.data_path = './cifar-100'

if args.dataset == 'fashionmnist' or args.dataset == 'mnist' :
    img_mean = torch.tensor(IMG_MEAN).view(1, 1).cuda()
    img_std = torch.tensor(IMG_STD).view(1, 1).cuda()
else:
    img_mean = torch.tensor(IMG_MEAN).view(1, 3, 1, 1).cuda()
    img_std = torch.tensor(IMG_STD).view(1, 3, 1, 1).cuda()

print(args)

stats_log=open('./checkpoint/%s_%.2f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.2f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w') 

# %%
loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=12,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.2f_%s.pt'%(args.data_path,args.r,args.noise_mode))

# %%
print('| Building net')
net1, vae_model1 = create_model()
net2, vae_model2 = create_model()

pretrain_dir = './pretrains/%s/checkpoint_%s_%.2f'%(args.dataset, args.noise_mode, args.r)+'.tar'
checkpoint = torch.load(pretrain_dir)
epoch = checkpoint['epoch']
print('loading checkpoints at %d' % epoch)
net1.load_state_dict(checkpoint['net1_state_dict'])
net2.load_state_dict(checkpoint['net2_state_dict'])

cudnn.benchmark = True

# %%
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_vae1 = optim.Adam(vae_model1.parameters(), lr=args.vae_lr)
optimizer_vae2 = optim.Adam(vae_model2.parameters(), lr=args.vae_lr)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

# %%
def warmup_vae(net, vae_model,vae_optimizer,train_all_loader, all_dataset, net_1):
    net.eval()
    vae_model.train()
    num_iter = (len(train_all_loader.dataset)//train_all_loader.batch_size)+1
    for batch_idx, (vae_data, clean_label, vae_targets, index) in enumerate(train_all_loader):
        vae_optimizer.zero_grad()
        if args.noise_mode != 'instance' and args.noise_mode != 'dependent':
            gt_t = all_dataset.t
            first_dim_indices = np.arange(vae_data.shape[0])
            gt_t = np.repeat(gt_t[np.newaxis, :, :], vae_data.shape[0], axis=0)
            gt_t = gt_t[first_dim_indices,clean_label,:]
        else:
            gt_t = all_dataset.t[index]
        vae_loss, recons_loss, nce_loss, kld_loss, L1 = train_vae(vae_data, vae_targets, gt_t, net, vae_model, epoch, batch_idx, net_1)
        loss = vae_loss + args.lambda_M * L1
        loss.backward()  
        vae_optimizer.step()
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t  loss: %.4f  Recons loss: %.2f  Noisy CE loss: %.2f  KL loss: %.2f  L1 loss: %.2f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item(), recons_loss.item(), nce_loss.item(), kld_loss.item(), L1.item()))
        sys.stdout.flush()

# %%
temp_ = loader.run('warmup')
img, target, _ = next(iter(temp_))

# %%
def my_vae_loss(x, n_logits, data, targets, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v):
    recons_loss = F.mse_loss(x, data, reduction="mean")
    ce_loss = CEloss(n_logits, targets)
    kld_1 = -0.5 * (1 + z_logvar - p_z_v - (1/p_z_v.exp()+1e-7) * (z_mean - p_z_m) ** 2 - (z_logvar-p_z_v).exp())
    kld_1 = torch.mean(kld_1, dim=2)
    kld_2 = -0.5 * (1 + log_var_s - mu_s ** 2 - (log_var_s).exp())
    kld_loss = torch.mean(torch.sum(kld_1, dim = 1), dim = 0) + torch.mean(torch.sum(kld_2, dim = 1), dim = 0)

    return recons_loss+ce_loss+kld_loss, recons_loss, ce_loss, kld_loss

# %%
warmup_trainloader = loader.run('warmup')
test_loader = loader.run('test')
eval_loader = loader.run('eval_train')

test(0,net1,net2)

start = time.time()
epoch = 0
pbar = tqdm(desc = 'Epochs', total = args.num_epochs)
while epoch < args.num_epochs:   
    lr=args.lr
    if epoch >= decay_epoch:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    lr=args.vae_lr
    if epoch >= decay_epoch:
        lr /= 10      
    for param_group in optimizer_vae1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_vae2.param_groups:
        param_group['lr'] = lr

    train_all_loader, all_dataset = loader.run("all")
    
    if epoch < warm_up:
        print('Warmup VAE1')
        warmup_vae(net1,vae_model1,optimizer_vae1,train_all_loader,all_dataset,net_1=True)
        print('\nWarmup VAE2')
        warmup_vae(net2,vae_model2,optimizer_vae2,train_all_loader,all_dataset,net_1=False)
    else:
        prob1,all_loss[0],eval_correct1=eval_train(net1,all_loss[0])   
        prob2,all_loss[1],eval_correct2=eval_train(net2,all_loss[1])            
        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)
        print('Train Net1')
        print('updating loader')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        loss_1 = train(epoch,net1,net2,optimizer1,vae_model1,vae_model2,optimizer_vae1,labeled_trainloader, unlabeled_trainloader, train_all_loader, all_dataset, net_1=True) # train net1  
        
        print('\nTrain Net2')
        print('updating loader')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        loss_2 = train(epoch,net2,net1,optimizer2,vae_model2,vae_model1, optimizer_vae2,labeled_trainloader, unlabeled_trainloader, train_all_loader, all_dataset, net_1=False) # train net2     
    test(epoch,net1,net2)
    pbar.update(epoch)
    epoch += 1
    torch.save({
            'epoch': epoch,
            'net1_state_dict': net1.state_dict(),
            'net2_state_dict': net2.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'vae_model1_state_dict': vae_model1.state_dict(),
            'vae_model2_state_dict': vae_model2.state_dict(),
            'optimizer_vae1_state_dict': optimizer_vae1.state_dict(),
            'optimizer_vae2_state_dict': optimizer_vae2.state_dict(),
            }, './saved/%s/checkpoint_%s_%.2f'%(args.dataset, args.noise_mode, args.r)+'.tar')
pbar.close()
end = time.time()
print(end - start)
