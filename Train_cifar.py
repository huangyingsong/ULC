from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from torch.nn import Parameter
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--tau_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='MC dropout rate')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.3, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imbalance', default=0.1, type=float)
parser.add_argument('--method', default='div_each_2', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

MC_Nsamples = 10

def safeMakeDir(tdir):
    if not os.path.isdir(tdir):
        os.mkdir(tdir)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, labels_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, labels_u = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)
        batch_size_u = inputs_u.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        labels_u = torch.zeros(batch_size_u, args.num_class).scatter_(1, labels_u.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, labels_u = inputs_u.cuda(), inputs_u2.cuda(), labels_u.cuda()

        with torch.no_grad():
            outputs_u11, _ = net(inputs_u).split(args.num_class, 1)
            outputs_u12, _ = net(inputs_u2).split(args.num_class, 1)
            outputs_u21, _ = net2(inputs_u).split(args.num_class, 1)
            outputs_u22, _ = net2(inputs_u2).split(args.num_class, 1) 
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x, _ = net(inputs_x).split(args.num_class, 1)
            outputs_x2, _ = net(inputs_x2).split(args.num_class, 1)
            
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

        labels_x, labels_u = labels_x.float(), labels_u.float()

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        outputs = net(mixed_input)
        logits, sigma = outputs.split(args.num_class, 1)
        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma, min=0, max=10)

        with torch.no_grad():
            net.c.data = torch.clamp(net.c.data, min=0, max=10) 

        prob_total = torch.zeros((MC_Nsamples, outputs.size(0), args.num_class))
        for t in range(MC_Nsamples):
            epsilon_s = torch.randn(sigma.size())
            epsilon_s = epsilon_s.cuda()
            epsilon_s = torch.mul(sigma, epsilon_s)

            epsilon_c = torch.randn(net.c.size())
            epsilon_c = epsilon_c.cuda()
            epsilon_c = torch.mul(net.c, epsilon_c)
            epsilon_x = torch.mm(logits, epsilon_c)
            logits_noisy = logits + epsilon_x + epsilon_s
            prob_total[t] = F.softmax(logits_noisy, dim=1)

        prob_ave = torch.mean(prob_total, 0)
        prob_ave = prob_ave.cuda()
        prob_ave += 1e-12
        
        prob_ave_x = prob_ave[:batch_size*2]
        prob_ave_u = prob_ave[batch_size*2:]

        Lx, Lu, lamb = criterion(prob_ave_x, mixed_target[:batch_size*2], prob_ave_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)

        

        loss = Lx + lamb * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.4f  Unlabeled loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, clean_labels, path) in enumerate(dataloader):      
        # Transform label to one-hot
        labels = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1,1), 1)        
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               

        mu, sigma = outputs.split(args.num_class, 1)
        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma, min=0, max=10)

        prob_total = torch.zeros((MC_Nsamples, outputs.size(0), args.num_class))
        for t in range(MC_Nsamples):
            epsilon = torch.randn(sigma.size())
            epsilon = epsilon.cuda()
            logits_noisy = mu + torch.mul(sigma, epsilon)
            prob_total[t] = F.softmax(logits_noisy, dim=1)
        prob_ave = torch.mean(prob_total, 0)
        prob_ave = prob_ave.cuda()
        prob_ave += 1e-12

        loss = -torch.mean(torch.sum(torch.log(prob_ave) * labels, dim=1))

        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(prob_ave)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def f1_score(output, target, numclass=10):
    _, maxk = output.max(dim=1,keepdim=False)
    label_one_hot = nn.functional.one_hot(target, numclass).float()
    pred_one_hot = nn.functional.one_hot(maxk, numclass).float()
    tp = label_one_hot * pred_one_hot
    tp = tp.sum(dim=0)
    fn = label_one_hot * (1 - pred_one_hot)
    fn = fn.sum(dim=0)
    fp = (1 - label_one_hot) * pred_one_hot
    fp = fp.sum(dim=0)
    tn = (1 - label_one_hot) * (1 - pred_one_hot)
    tn = tn.sum(dim=0)
    return tp, fp, fn, tn

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    device = torch.device('cuda')
    TP = torch.zeros(args.num_class,device=device)
    FP = torch.zeros(args.num_class,device=device)
    FN = torch.zeros(args.num_class,device=device)
    TN = torch.zeros(args.num_class,device=device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs).split(args.num_class, 1)
            outputs2, _ = net2(inputs).split(args.num_class, 1)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
            tp, fp, fn, tn = f1_score(outputs, targets, numclass=args.num_class)
            TP += tp
            FP += fp
            FN += fn
            TN += tn

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('\n| Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))

    print("\n| Recall:\n", Recall)
    test_log.write('Recall:\n')
    test_log.write(str(Recall))
    print("\n| Precision:\n", Precision)
    test_log.write('\nPrecision:\n')
    test_log.write(str(Precision) + '\n')
    test_log.flush()  

def eval_train_each_2(model,all_loss,ep,Nsamples=10):
    model.train()
    
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    losses = torch.zeros(Nsamples, NUM_SAMPLES)
    aps = torch.zeros(Nsamples, NUM_SAMPLES, args.num_class)
    sigmas = torch.zeros(Nsamples, NUM_SAMPLES, args.num_class)
    p_losses = torch.zeros(Nsamples, NUM_SAMPLES)
    all_targets = torch.zeros(NUM_SAMPLES)    
    all_clean_targets = torch.zeros(NUM_SAMPLES)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, clean_targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            b = inputs.size(0)
            for i in range(Nsamples):
                outputs = model(inputs) 
                mu, sigma = outputs.split(args.num_class, 1)
                loss = CE(mu, targets)  
                p = SM(mu)  
                p_star = p[range(p.size(0)), targets]

                t_onehot = nn.functional.one_hot(targets, args.num_class).float()
                p_ = p - t_onehot
                p_max = p_.max(dim=1, keepdim=False)[0]
                p_minus = p_star - p_max
                p_loss = (p_minus + 1.) / 2

                losses[i, index] = loss.cpu()
                p_losses[i, index] = p_loss.cpu()
                aps[i, index] = p.cpu()
                sigmas[i, index] = sigma.cpu()
        
            all_targets[index] = targets.cpu().float()
            all_clean_targets[index] = clean_targets.cpu().float()
    all_loss.append(losses)
    if len(all_loss) > 5:
        all_loss = all_loss[-5:]

    if args.r>=0.9: # average loss over last 5 epochs to improve convergence stability
        if ep < 50:
            history = torch.cat(all_loss)
        else:
            history = all_loss[-1]
        losses = history
    else:
        losses = losses
   
    mean_loss = losses.mean(dim=0)
    loss_c_min = mean_loss.min()
    loss_c_max = mean_loss.max()

    # fit a two-component GMM to each loss
    num_class = args.num_class
    gmm_w = np.zeros((num_class, 2))
    gmm_m = np.zeros((num_class, 2))
    all_prob = np.zeros(NUM_SAMPLES)
    all_pred = np.zeros((NUM_SAMPLES), dtype=np.int8)

    mapss = np.ones(num_class) / num_class
    mentropy = - mapss * np.log(mapss)
    mentropy = mentropy.sum()

    mean_ap = aps.mean(dim=0)
    prd_entropy = - mean_ap * torch.log(mean_ap + 1e-10)
    prd_entropy = prd_entropy.sum(dim=1)
    prd_entropy = (prd_entropy - prd_entropy.min()) / (mentropy - prd_entropy.min())
    prd_entropy /= 1.

    p_sign = (p_losses.mean(dim=0) < 0.5)
    prd_entropy[p_sign] = 0.9

    pr = 0.1
    for c in range(num_class):
        c_idx_bool = (all_targets == c)
        c_idx = c_idx_bool.nonzero() 
        c_idx = c_idx.squeeze()
        loss_c_N = losses[:, c_idx]

        loss_c = loss_c_N.mean(dim=0)
        loss_c_min = loss_c.min()
        loss_c_max = loss_c.max()
        loss_c = (loss_c - loss_c_min)/(loss_c_max - loss_c_min)
        loss_c = 1. - loss_c

        prd_entropy_c = prd_entropy[c_idx]
        prd_entropy_c = prd_entropy_c.numpy()

        loss_c = loss_c.reshape(-1, 1)
        loss_c = loss_c.numpy()

        gmm = GaussianMixture(n_components=2,max_iter=100,tol=1e-2,reg_covar=5e-4)
        gmm.fit(loss_c)
        sortidx = gmm.means_.squeeze().argsort()
        prob_all = gmm.predict_proba(loss_c) 
        prob = prob_all[:,gmm.means_.argmax()]         
        
        id_min = prob.argmin()
        id_neg = (loss_c < loss_c[id_min])
        id_neg = id_neg.squeeze()
        id_neg = id_neg.nonzero()
        prob[id_neg] = prob.min()
        id_max = prob.argmax()
        id_pos = (loss_c > loss_c[id_max])
        id_pos = id_pos.squeeze()
        id_pos = id_pos.nonzero()
        prob[id_pos] = prob.max()

        proba = 1 - prob

        prob_all[:, sortidx[0]] = proba
        prob_all[:, sortidx[1]] = prob

        gmm_w[c] = np.asarray(gmm.weights_).squeeze()
        gmm_m[c] = np.asarray(gmm.means_).squeeze()


        uncertainty_prob = (1 - prd_entropy_c)

        prd_combine_c = (prob ** (1 - pr)) * (uncertainty_prob ** pr)
        prd_combine_n = (proba ** (1 - pr)) * (uncertainty_prob ** pr)
        pred_clean = (prd_combine_c >= args.tau_threshold)
        pred_noisy = (prd_combine_n > 1 - args.tau_threshold)
        pred = pred_clean.astype(np.int8) - pred_noisy.astype(np.int8)

        all_prob[c_idx] = prd_combine_c
        all_pred[c_idx] = pred

    return all_prob, all_loss, all_pred

def eval_train(model,all_loss,ep):    
    model.eval()
    losses = torch.zeros(NUM_SAMPLES)    
    all_targets = torch.zeros(NUM_SAMPLES)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
                all_targets[index[b]] = targets[b]
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    if len(all_loss) > 5:
        all_loss = all_loss[-5:]

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
    pred = (prob > args.tau_threshold)      

    epoch = ep
    loss_f = OutDir + '/stat_' + str(epoch) + '.pkl'
    data = {}
    data['prob'] = input_loss
    data['targets'] = all_targets
    with open(loss_f, 'wb') as f:
        pickle.dump(data, f)
    print('gmm mean: ', gmm.means_)

    return prob,all_loss,pred

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = outputs_u

        Lx = -torch.mean(torch.sum(torch.log(outputs_x) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = outputs
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(dropout_rate=0.5):
    model = ResNet18(num_classes=args.num_class * 2)
    model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, args.num_class * 2, bias=True)
    )
    model = model.cuda()
    model.c = Parameter(torch.zeros(args.num_class, args.num_class).cuda() + 1e-3)
    return model


CheckpointDir = './checkpoint'
safeMakeDir(CheckpointDir)
OutDir = './output'
safeMakeDir(OutDir)

if args.dataset=='cifar10':
    warm_up = 30
elif args.dataset=='cifar100':
    warm_up = 40

if args.imbalance < 1.:
    stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_imbalance_stats.txt','w') 
    test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_imbalance_acc.txt','w')     
    nf = '%s/%.1f_%s_imbalance.json'%(args.data_path,args.r,args.noise_mode)
    NUM_SAMPLES = int(50000 * (1. + args.imbalance) / 2.)
else:
    stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
    test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
    nf = '%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode)
    NUM_SAMPLES = 50000

loader = dataloader.cifar_dataloader(args.dataset,imbalance=args.imbalance,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file=nf)

print('| Building net')
net1 = create_model(args.dropout_rate)
net2 = create_model(args.dropout_rate)
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
SM = nn.Softmax(dim=1)
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) 
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1)
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    if args.method=='div_all':
        prob1,all_loss[0], pred1 = eval_train(net1,all_loss[0], epoch)   
        prob2,all_loss[1], pred2 = eval_train(net2,all_loss[1], epoch)          
    if args.method=='div_each_2':
        prob1,all_loss[0], pred1 = eval_train_each_2(net1,all_loss[0], epoch)   
        prob2,all_loss[1], pred2 = eval_train_each_2(net2,all_loss[1], epoch)          
               
    test(epoch,net1,net2)  


