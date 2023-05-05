import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time

def get_now_time():
    """
    获取当前日期时间
    :return:当前日期时间
    """
    now =  time.localtime()
    now_time = time.strftime("%Y-%m-%d_%H-%M-%S", now)
    # now_time = time.strftime("%Y-%m-%d ", now)
    return now_time

class Config(object):

    """配置参数"""
    def __init__(self):
        self.device = torch.device('cuda')                          # 设备
        self.dropout = 0.2                                          # 随机失活
        self.num_classes = 5                                        # 类别数
        self.input_dim = 300                                        # 输入向量维度

        self.num_epochs = 50                                           # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.learning_rate = 1e-4                                       # 学习率
        
        self.hidden_layer = [512,1024,2048,2048,1024,512,128,32]              # 隐藏层结构

        nowTime = get_now_time()
        self.model_name = 'output/NN_'+nowTime

class Model(nn.Module):

    def __init__(self,config):
        super(Model, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_layer = config.hidden_layer
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        

        self.in_layer = nn.Linear(self.input_dim,self.hidden_layer[0])

        module_list = []
        for i in range(len(self.hidden_layer)-1):
            module_list.append(nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1]))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(self.dropout))

        self.Linear_Hidden_Layer = nn.ModuleList(module_list)
        self.fc = nn.Linear(self.hidden_layer[-1],self.num_classes)

    def forward(self, data):
        # data shape : (batch_size,input_dim)
        x = self.in_layer(data)

        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.Linear_Hidden_Layer):
            #print(i,l)
            x = l(x)
        
        x = self.fc(x)

        return x
        

if __name__ == '__main__':
    # 使用torchsummary打印模型信息

    config = Config()
    M = Model(config).to(config.device)

    from torchsummary import summary

    summary(M,(100,config.input_dim))
