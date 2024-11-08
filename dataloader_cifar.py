from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import torchvision
import tools
import pandas as pd
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
                num_classes_ = 10
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']          
                num_classes_ = 100
            elif dataset=='fashionmnist':
                self.test_data = np.load('./datasets/fashionmnist/test_images.npy')
                self.test_label = np.load('./datasets/fashionmnist/test_labels.npy')   
                num_classes_ = 10
            elif dataset=='mnist':
                self.test_data = np.load('./datasets/mnist/test_images.npy')
                self.test_label = np.load('./datasets/mnist/test_labels.npy') - 1 # 0-9
                num_classes_ = 10
            elif dataset=='svhn':
                self.test_data = np.load('./datasets/svhn/test_images.npy')
                self.test_label = np.load('./datasets/svhn/test_labels.npy')       
                num_classes_ = 10
                self.test_data = self.test_data.reshape((-1, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))         
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                num_classes_ = 10
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
                feature_size = 32*32*3
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                num_classes_ = 100
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
                feature_size = 32*32*3
            elif dataset=='fashionmnist':
                train_data = np.load('./datasets/fashionmnist/train_images.npy')
                train_label = np.load('./datasets/fashionmnist/train_labels.npy')
                iter_dataset = zip(train_data, train_label)
                num_classes_ = 10
                feature_size = 28*28
            elif dataset=='mnist':
                train_data = np.load('./datasets/mnist/train_images.npy')
                train_label = np.load('./datasets/mnist/train_labels.npy')
                iter_dataset = zip(train_data, train_label)
                num_classes_ = 10
                feature_size = 28*28
            elif dataset=='svhn':
                train_data = np.load('./datasets/svhn/train_images.npy')
                train_label = np.load('./datasets/svhn/train_labels.npy')
                iter_dataset = zip(train_data, train_label)
                train_data = train_data.reshape((-1, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
                num_classes_ = 10
                feature_size = 32*32*3
            self.clean_labels = train_label.copy()

            if noise_mode in ['worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']:
                noise_label = torch.load('./CIFAR-N/CIFAR-10_human.pt') 
                worst_label = noise_label['worse_label'] 
                aggre_label = noise_label['aggre_label'] 
                random_label1 = noise_label['random_label1'] 
                random_label2 = noise_label['random_label2'] 
                random_label3 = noise_label['random_label3']
                print('loading %s'%(noise_mode))
                noise_label = noise_label[noise_mode]
                self.t = np.ones((noise_label.shape[0], 10))
            elif noise_mode == 'noisy_label':
                noise_label = torch.load('./CIFAR-N/CIFAR-100_human.pt') 
                print('loading %s'%(noise_mode))
                noise_label = noise_label[noise_mode]
                self.t = np.ones((noise_label.shape[0], 100))
            elif noise_mode == 'dependent':
                noise_label = torch.Tensor(pd.read_csv(os.path.join('./datasets/%s/label_noisy'%dataset, noise_mode+str(self.r)+'.csv'))['label_noisy'].values.astype(int)).long()
                print(noise_label[:10])
                softmax_out_avg = np.load('./datasets/%s/label_noisy/'%dataset+'softmax_out_avg_%s.npy'%str(self.r))
                print(softmax_out_avg.shape)
                self.t = softmax_out_avg
            else:
                if os.path.exists(noise_file):
                    print('loading %s'%noise_file)
                    noise_label = torch.load(noise_file)
                    # print(noise_label[:10])
                    self.t = np.load(noise_file+'_selected_t.npy')
                else:
                    data_ = torch.from_numpy(train_data).float().cuda()
                    targets_ = torch.IntTensor(train_label).cuda()
                    dataset = zip(data_, targets_)
                    if noise_mode == 'instance':
                        train_label = torch.FloatTensor(train_label).cuda()
                        noise_label, self.t = tools.get_instance_noisy_label(self.r, dataset, train_label, num_classes = num_classes_, feature_size = feature_size, norm_std=0.1, seed=123)
                    elif noise_mode == 'sym':
                        noise_label = []
                        idx = list(range(train_data.shape[0]))
                        random.shuffle(idx)
                        num_noise = int(self.r*train_data.shape[0])            
                        noise_idx = idx[:num_noise]
                        for i in range(train_data.shape[0]):
                            if i in noise_idx:
                                noiselabel = random.randint(0,num_classes_-1)
                                noise_label.append(noiselabel)
                            else:    
                                noise_label.append(train_label[i])   
                        noise_label = np.array(noise_label)
                        P = np.ones((num_classes_, num_classes_))
                        n = self.r * (num_classes_-1)/num_classes_
                        P = (n / (num_classes_ - 1)) * P
                        for i in range(0, num_classes_):
                            P[i, i] = 1. - n
                        self.t=P
                    elif noise_mode == 'pair':
                        train_label = np.array(train_label)
                        train_label = train_label.reshape((-1,1))
                        noise_label = tools.noisify_pairflip(train_label, self.r, 123, num_classes_)
                        noise_label = noise_label[:, 0]
                        P = np.ones((num_classes_, num_classes_))
                        n = self.r
                        P[0, 0], P[0, 1] = 1. - n, n
                        for i in range(1, num_classes_-1):
                            P[i, i], P[i, i + 1] = 1. - n, n
                        P[num_classes_-1, num_classes_-1], P[num_classes_-1, 0] = 1. - n, n
                        self.t=P
                    print("save noisy labels to %s ..."%noise_file)     
                    torch.save(noise_label, noise_file)   
                    np.save(noise_file+'_selected_t.npy', self.t)
            
            if self.mode == 'warmup':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(self.clean_labels))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    print('Numer of labeled samples:%d   AUC:%.3f'%(pred.sum(),auc))
                    print('Precision: ', clean[pred_idx].sum()/pred_idx.shape[0])
                    print('Recall: ', clean[pred_idx].sum()/clean.sum())
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]              
                    self.probability = [probability[i] for i in pred_idx]                                         
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob  
        elif self.mode=='warmup':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='all':
            img, clean_label, target = self.train_data[index], self.clean_labels[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, clean_label, target, index  
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset in ['cifar10', 'svhn']:
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset == 'fashionmnist' or self.dataset == 'mnist':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(28, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081)),
                ])
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="warmup",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        
        elif mode=='all':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader, all_dataset
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, probability=prob)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='warmup', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        



if __name__ == '__main__':
    stats_log=open('./checkpoint/%s_%.2f_%s'%('cifar10', 0.5, 'instance')+'_stats.txt','w') 
    loader = cifar_dataloader('cifar10',r=0.5,noise_mode='instance',batch_size=64,num_workers=16,\
    root_dir='./cifar-10',log=stats_log,noise_file='%s/%.2f_%s.pt'%('./cifar-10',0.5,'instance'))
    train_loader = loader.run("all")
    warmup_loader = loader.run("warmup")
    for batch_idx, (inputs, labels, _) in enumerate(warmup_loader):
        torchvision.utils.save_image(inputs, './sample_imgs/samples_{}.png'.format(batch_idx), nrow=8, padding=8)
        if batch_idx>20:
            break
    
    for batch_idx, (inputs, labels, u, _) in enumerate(train_loader):
        torchvision.utils.save_image(inputs, './sample_imgs/samples_{}.png'.format(batch_idx), nrow=8, padding=8)
        if batch_idx>20:
            break