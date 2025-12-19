from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture
from torch.nn import Parameter
import pickle

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0.0, type=float, help='weight for unsupervised loss')
parser.add_argument('--tau_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--uncertainty_threshold', default=0.0, type=float, help='sample predictive uncertainty threshold')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='MC dropout rate')
parser.add_argument('--u_threshold', default=0.95, type=float, help='fixmatch tau')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='./data/clothing1M', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--method', default='div_each_2', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        batch_size_u = inputs_u.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        
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

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        outputs = net(mixed_input)
        logits, sigma = outputs.split(args.num_class, 1)
        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma, min=0, max=10)

        with torch.no_grad():
            net.c.data = torch.clamp(net.c.data, min=0, max=10)

        prob_total = torch.zeros((MC_Nsamples, outputs.size(0), args.num_class))
        for t in range(MC_Nsamples):
            epsilon_s = torch.randn(sigma.size())
            sigma = sigma + torch.mm(sigma, net.c)
            epsilon_s = epsilon_s.cuda()
            epsilon_s = torch.mul(sigma, epsilon_s)

            logits_noisy = logits + epsilon_s
            prob_total[t] = F.softmax(logits_noisy, dim=1)

        prob_ave = torch.mean(prob_total, 0)
        prob_ave = prob_ave.cuda()

        prob_ave += 1e-12

        
        Lx = -torch.mean(torch.sum(torch.log(prob_ave) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')

        sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f'
            %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))

        sys.stdout.flush()
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        # Transform label to one-hot
        labels = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1,1), 1)
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              

        mu, sigma = outputs.split(args.num_class, 1)
        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma, min=0, max=10)

        with torch.no_grad():
            net.c.data = torch.clamp(net.c.data, min=0, max=10)

        prob_total = torch.zeros((MC_Nsamples, outputs.size(0), args.num_class))
        for t in range(MC_Nsamples):
            epsilon_s = torch.randn(sigma.size())
            sigma = sigma + torch.mm(sigma, net.c)
            epsilon_s = epsilon_s.cuda()
            epsilon_s = torch.mul(sigma, epsilon_s)

            logits_noisy = mu + epsilon_s
            prob_total[t] = F.softmax(logits_noisy, dim=1)

        prob_ave = torch.mean(prob_total, 0)
        prob_ave = prob_ave.cuda()
        prob_ave += 1e-12

        loss = -torch.mean(torch.sum(torch.log(prob_ave) * labels, dim=1))
        
        penalty = conf_penalty(prob_ave)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
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

def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    device = torch.device('cuda')
    TP = torch.zeros(args.num_class,device=device)
    FP = torch.zeros(args.num_class,device=device)
    FN = torch.zeros(args.num_class,device=device)
    TN = torch.zeros(args.num_class,device=device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs).split(args.num_class, 1)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
            tp, fp, fn, tn = f1_score(outputs, targets, numclass=args.num_class)
            TP += tp
            FP += fp
            FN += fn
            TN += tn
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    print("\nRecall:\n", Recall)
    test_log.write('\nRecall:\n')
    test_log.write(str(Recall))
    print("\nPrecision:\n", Precision)
    test_log.write('\nPrecision:\n')
    test_log.write(str(Precision) + '\n')
    test_log.flush()  


    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = './checkpoint/%s_net%d.pth.tar'%(args.id,k)
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs).split(args.num_class, 1)
            outputs2, _ = net2(inputs).split(args.num_class, 1)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc    


def eval_train_each_2(model,ep,Nsamples=10):
    model.train()
    
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(Nsamples, num_samples)
    p_losses = torch.zeros(Nsamples, num_samples)
    aps = torch.zeros(Nsamples, num_samples, args.num_class)   
    sigmas = torch.zeros(Nsamples, num_samples, args.num_class)
    all_targets = torch.zeros(num_samples)    
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            b = inputs.size(0)
            for i in range(Nsamples):
                outputs, sigma = model(inputs).split(args.num_class, 1)
                loss = CE(outputs, targets)  
                p = SM(outputs)  
                p_star = p[range(p.size(0)), targets]

                t_onehot = nn.functional.one_hot(targets, args.num_class).float()
                p_ = p - t_onehot
                p_max = p_.max(dim=1, keepdim=False)[0]
                p_loss = p_star - p_max
                p_loss = (p_loss + 1.) / 2

                losses[i, n:(n+b)] = loss.cpu()
                p_losses[i, n:(n+b)] = p_loss.cpu()
                aps[i, n:(n+b)] = p.cpu()
                sigmas[i, n:(n+b)] = sigma.cpu()

            all_targets[n:(n+b)] = targets.cpu().float()
            paths += path
            n += b
            
    
    mean_loss = losses.mean(dim=0)
    loss_c_min = mean_loss.min()
    loss_c_max = mean_loss.max()
    # fit a two-component GMM to each loss
    num_class = args.num_class
    gmm_w = np.zeros((num_class, 2))
    gmm_m = np.zeros((num_class, 2))
    all_prob = np.zeros(num_samples)
    all_pred = np.zeros((num_samples), dtype=np.int8)

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
        loss_c = (loss_c - loss_c_min)/(loss_c_max - loss_c_min)
        loss_c = 1. - loss_c

        loss_c_N = (loss_c_N - loss_c_min)/(loss_c_max - loss_c_min)
        loss_c_N = 1. - loss_c_N

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


    return all_prob, all_pred, paths

def eval_train(model, ep):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       

    pred = (prob > args.tau_threshold)      

    return prob, pred, paths
    
def linear_rampup(current, warm_up=1, rampup_length=20):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class NegEntropy(object):
    def __call__(self,outputs, mask=None):
        probs = outputs
        if mask is None:
            return torch.mean(torch.sum(probs.log()*probs, dim=1))
        else:
            return torch.mean(torch.sum(probs.log()*probs, dim=1)*mask)
               
def create_model(dropout_rate=0.5):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(2048,args.num_class * 2, bias=True)
    )
    model = model.cuda()
    model.c = Parameter(torch.zeros(args.num_class, args.num_class).cuda() + 1e-3)
    return model     


CheckpointDir = './checkpoint'
safeMakeDir(CheckpointDir)
log=open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()
test_log=open('./checkpoint/test_%s.txt'%args.id,'w')     
test_log.flush()

OutDir = './output'
safeMakeDir(OutDir)

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model(args.dropout_rate)
net2 = create_model(args.dropout_rate)
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
SM = nn.Softmax(dim=1)
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
Replace_C = False
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    if epoch<1:     # warm up  
        print('\n| epoch:%d\n'%(epoch))
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        print('\n| epoch:%d\n'%(epoch))
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)              # train net2
    
    val_loader = loader.run('val') # validation
    test_log.write('validation Epoch:%d\n'%epoch)
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 
    print('net1 c: ', net1.c)
    print('net2 c: ', net2.c)

    if args.method=='div_all':
        print('\n==== net 1 evaluate next epoch training data loss ====') 
        eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
        prob1, pred1, paths1 = eval_train(net1, epoch) 
        print('\n==== net 2 evaluate next epoch training data loss ====') 
        eval_loader = loader.run('eval_train')  
        prob2, pred2, paths2 = eval_train(net2, epoch) 
    if args.method=='div_each_2':
        print('\n==== net 1 evaluate next epoch training data loss ====') 
        eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
        prob1, pred1, paths1 = eval_train_each_2(net1, epoch) 
        print('\n==== net 2 evaluate next epoch training data loss ====') 
        eval_loader = loader.run('eval_train')  
        prob2, pred2, paths2 = eval_train_each_2(net2, epoch) 

test_loader = loader.run('test')
net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar'%args.id))
net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar'%args.id))
acc = test(net1,net2,test_loader)      

log.write('Test Accuracy:%.2f\n'%(acc))
log.flush() 
