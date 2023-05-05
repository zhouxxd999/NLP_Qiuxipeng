import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

from tqdm import tqdm 

def evaluate_accuracy_gpu(model,data_iter,
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """使用GPU计算模型再数据集上面的精度"""
    acc_sum , n = 0.0,0
    loss_total = 0
    model.eval() #评估模式
    with torch.no_grad():
        for X,y in data_iter:
            
            pr = model(X.to(device))

            loss = F.cross_entropy(pr, y.to(device))
            loss = loss.cpu()
            loss_total += loss
            acc_sum += (pr.argmax(dim=1) ==
                        y.to(device)).float().sum().cpu().item()
            model.train() #改回训练模式
            #y.shape[0]为一个batch的样本数
            n += y.shape[0]
    model.train() #训练模式
    return acc_sum/n,loss_total/len(data_iter)

def train(config, model, train_iter, dev_iter, test_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #total_batch = 0  # 记录进行到多少batch
    dev_best_loss = np.inf
    
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        batch_count = 0
        train_loss_sum  =0.0
        for trains, labels in tqdm(train_iter):
            
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            outputs = model(trains)
         
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu()
            batch_count = batch_count + 1
        train_acc,train_loss = evaluate_accuracy_gpu(model,train_iter)
        dev_acc,dev_loss = evaluate_accuracy_gpu(model,dev_iter)
        print('train loss : %.4f ,train acc:%.3f , dev loss : %.4f,dev acc : %.3f '%(train_loss,train_acc,dev_loss,dev_acc))
        if dev_loss < dev_best_loss :
            dev_best_loss = dev_loss
            print('saving model ...')
            time.sleep(1)
            torch.save(model.state_dict(), config.model_name)


