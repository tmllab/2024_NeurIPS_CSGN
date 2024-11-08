from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from models.InceptionResNetV2 import *
from models.vae import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=8, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--vae_lr', '--vae_learning_rate', default=0.001, type=float, help='initial vae learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--lambda_elbo', default=0.1, type=float, help='weight for elbo')
parser.add_argument('--lambda_M', default=0.001, type=float, help='weight for M')
parser.add_argument('--z_dim', default=4, type=int)
parser.add_argument('--data_path', default='/home/kh31/webvision/', type=str, help='path to dataset')
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()

print(args)

local_rank = args.local_rank

# DDP init
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl backend
rank = torch.distributed.get_rank()
random.seed(args.seed+rank)
torch.manual_seed(args.seed+rank)
torch.cuda.manual_seed_all(args.seed+rank)


# Training
def train(epoch,net,net2,optimizer,vae_model, vae_model2,optimizer_vae,labeled_trainloader,unlabeled_trainloader, train_all_loader, GLOBAL_STEPS):
    net.train()
    net2.eval() #fix one network and train the other
    vae_model.train()
    vae_model2.eval()
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    all_train_iter = iter(train_all_loader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)

        if GLOBAL_STEPS < WARMUP_STEPS:
            lr = warmup_lr_func(GLOBAL_STEPS, args.lr)
            for param in optimizer.param_groups:
                param['lr']=lr
            lr = warmup_lr_func(GLOBAL_STEPS, args.vae_lr)
            for param in optimizer_vae.param_groups:
                param['lr']=lr
            GLOBAL_STEPS+=1
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

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
            px = w_x*labels_x + (1-w_x)*px              
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

        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss_dm = Lx + penalty
        try:
            vae_data, vae_targets, pred_clean, _ = all_train_iter.next()
        except:
            all_train_iter = iter(train_all_loader)
            vae_data, vae_targets, pred_clean, _ = all_train_iter.next()
        
        vae_loss, recons_loss, nce_loss, kld_loss, L1 = train_vae(vae_data, vae_targets, pred_clean, net, vae_model)

        loss = loss_dm + args.lambda_elbo * vae_loss + args.lambda_M * L1
        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f  Recons loss: %.2f  Noisy CE loss: %.2f  KL loss: %.2f  L1 loss: %.2f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), recons_loss.item(), nce_loss.item(), kld_loss.item(), L1.item()))
        sys.stdout.flush()
    
    return GLOBAL_STEPS

# Train vae
def train_vae(vae_data, vae_targets, pred_clean, net, vae_model):
    vae_model.train()
    data = vae_data.cuda()
    targets = vae_targets.cuda()
    pred_clean = pred_clean.cuda()
    #forward
    x, n_logits, c_logits, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v, M_x, M_y  = vae_model(data, net)
    L1 = torch.norm(M_x, p=1) + torch.norm(M_y, p=1)
    # calculate loss
    vae_loss, recons_loss, nce_loss, kld_loss = my_vae_loss(x, n_logits, data, targets, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v)

    return vae_loss, recons_loss, nce_loss, kld_loss, L1


def my_vae_loss(x, n_logits, data, targets, z_mean, z_logvar, mu_s, log_var_s, p_z_m, p_z_v):
    recons_loss = F.mse_loss(x, data, reduction="mean")
    ce_loss = CEloss(n_logits, targets)
    kld_1 = -0.5 * (1 + z_logvar - p_z_v - (1/p_z_v.exp()+1e-7) * (z_mean - p_z_m) ** 2 - (z_logvar-p_z_v).exp())
    kld_1 = torch.mean(kld_1, dim=2)
    kld_2 = -0.5 * (1 + log_var_s - mu_s ** 2 - (log_var_s).exp())
    kld_loss = torch.mean(torch.sum(kld_1, dim = 1), dim = 0) + torch.mean(torch.sum(kld_2, dim = 1), dim = 0)

    return recons_loss+ce_loss+kld_loss, recons_loss, ce_loss, kld_loss

        
def warmup_vae(net, vae_model,vae_optimizer,dataloader):
    net.eval()
    vae_model.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(local_rank), labels.to(local_rank)
        vae_optimizer.zero_grad()
        pred_clean = torch.ones_like(labels)
        vae_loss, recons_loss, nce_loss, kld_loss, L1 = train_vae(inputs, labels, pred_clean, net, vae_model)
        loss = vae_loss + args.lambda_M * L1
        loss.backward()  
        vae_optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t  loss: %.4f  Recons loss: %.2f  Noisy CE loss: %.2f  KL loss: %.2f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item(), recons_loss.item(), nce_loss.item(), kld_loss.item()))
        sys.stdout.flush()
        
def test(epoch,net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)                 
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    return accs


def eval_train(model,all_loss):    
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]       
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' %(batch_idx,num_iter)) 
            sys.stdout.flush()    
                                    
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = InceptionResNetV2(num_classes=args.num_class)
    vae_model = VAE_WEBVISION(z_dim=args.z_dim, num_classes=args.num_class)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    vae_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae_model).to(local_rank)
    return model, vae_model

stats_log=open('./checkpoint/%s'%(args.id)+'_stats.txt','w') 
test_log=open('./checkpoint/%s'%(args.id)+'_acc.txt','w')     

warm_up=5

loader = dataloader.webvision_ddp_dataloader(batch_size=args.batch_size,num_workers=2,root_dir=args.data_path,log=stats_log, num_class=args.num_class)

print('| Building net')
net1, vae_model1 = create_model()
net2, vae_model2 = create_model()
cudnn.benchmark = True

if dist.get_rank() == 0:
    pretrain_dir = './pretrains/webvision/webvision_checkpoint_last.pth.tar'
    checkpoint = torch.load(pretrain_dir)
    epoch = checkpoint['epoch']
    print('loading checkpoints at %d' % epoch)
    net1.load_state_dict(checkpoint['net1_state_dict'])
    net2.load_state_dict(checkpoint['net2_state_dict'])

if local_rank != -1:
    net1 = DDP(net1, device_ids=[local_rank], output_device=local_rank)
    net2 = DDP(net2, device_ids=[local_rank], output_device=local_rank)
    vae_model1 = DDP(vae_model1, device_ids=[local_rank], output_device=local_rank)
    vae_model2 = DDP(vae_model2, device_ids=[local_rank], output_device=local_rank)

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_vae1 = optim.Adam(vae_model1.parameters(), lr=args.vae_lr, weight_decay=5e-4)
optimizer_vae2 = optim.Adam(vae_model2.parameters(), lr=args.vae_lr, weight_decay=5e-4)


CE = nn.CrossEntropyLoss(reduction='none').to(local_rank)
CEloss = nn.CrossEntropyLoss().to(local_rank)
conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
epoch=0

WARMUP_STEPS=258 * 3
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS1=0
GLOBAL_STEPS2=0

def warmup_lr_func(step, lr):
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor

    return float(lr)

while epoch < args.num_epochs:
    lr=args.lr
    if epoch >= 40:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr      
    lr=args.vae_lr
    if epoch >= 40:
        lr /= 10      
    for param_group in optimizer_vae1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_vae2.param_groups:
        param_group['lr'] = lr        
    
    eval_loader = loader.run('eval_train')  
    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')   
    
    if epoch<warm_up:   
        warmup_trainloader = loader.run('warmup')
        warmup_trainloader.sampler.set_epoch(epoch)
        print('\nWarmup VAE1')
        warmup_vae(net1, vae_model1,optimizer_vae1,warmup_trainloader)
        warmup_trainloader = loader.run('warmup')
        warmup_trainloader.sampler.set_epoch(epoch)
        print('\nWarmup VAE2')
        warmup_vae(net2, vae_model2,optimizer_vae2,warmup_trainloader)
        if dist.get_rank() == 0:
            torch.save({
                'epoch': epoch,
                'net1_state_dict': net1.module.state_dict(),
                'net2_state_dict': net2.module.state_dict(),
                'vae1_state_dict': vae_model1.module.state_dict(),
                'vae2_state_dict': vae_model2.module.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                'optimizer_vae1_state_dict': optimizer_vae1.state_dict(),
                'optimizer_vae2_state_dict': optimizer_vae2.state_dict()
                }, './webvision_checkpoint_warmup.pth.tar')
    else:   
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold) 
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train_all_loader = loader.run("all",pred2,prob2)
        labeled_trainloader.sampler.set_epoch(epoch)
        unlabeled_trainloader.sampler.set_epoch(epoch)
        train_all_loader.sampler.set_epoch(epoch)
        GLOBAL_STEPS1 = train(epoch,net1,net2,optimizer1,vae_model1,vae_model2,optimizer_vae1,labeled_trainloader, unlabeled_trainloader, train_all_loader, GLOBAL_STEPS1) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train_all_loader = loader.run("all",pred1,prob1)
        labeled_trainloader.sampler.set_epoch(epoch)
        unlabeled_trainloader.sampler.set_epoch(epoch)
        train_all_loader.sampler.set_epoch(epoch)
        GLOBAL_STEPS2 = train(epoch,net2,net1,optimizer2,vae_model2,vae_model1, optimizer_vae2,labeled_trainloader, unlabeled_trainloader, train_all_loader, GLOBAL_STEPS2) # train net2    

    
    web_acc = test(epoch,net1.module,net2.module,web_valloader)  
    imagenet_acc = test(epoch,net1.module,net2.module,imagenet_valloader)  
    if epoch<warm_up or epoch>15:          
        if dist.get_rank() == 0:
            print('\n==== net 1 evaluate training data loss ====') 
            prob1,all_loss[0]=eval_train(net1.module,all_loss[0])   
            np.save('prob1.npy', prob1)
            print('\n==== net 2 evaluate training data loss ====') 
            prob2,all_loss[1]=eval_train(net2.module,all_loss[1])
            np.save('prob2.npy', prob2)
            torch.save(all_loss,'./checkpoint/%s.pth.tar'%(args.id))        
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            prob1 = np.load('prob1.npy')
            prob2 = np.load('prob2.npy')
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()  
    if dist.get_rank() == 0:
        torch.save({
                'epoch': epoch+1,
                'all_loss': all_loss,
                'net1_state_dict': net1.module.state_dict(),
                'net2_state_dict': net2.module.state_dict(),
                'vae1_state_dict': vae_model1.module.state_dict(),
                'vae2_state_dict': vae_model2.module.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                'optimizer_vae1_state_dict': optimizer_vae1.state_dict(),
                'optimizer_vae2_state_dict': optimizer_vae2.state_dict()
                }, './webvision_checkpoint_last.pth.tar')
    epoch+=1
