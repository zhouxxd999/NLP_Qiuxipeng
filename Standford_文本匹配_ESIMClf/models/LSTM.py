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
        self.dropout = 0.1                                          # 随机失活
        self.num_classes = 5                                        # 类别数
        self.input_dim = 300                                        # 输入向量维度
        self.hidden_dim = 300                                       # 隐藏层特征维度
        self.hidden_layers = 3                                      # LSTM隐藏层个数
        self.BiDirection = True                                    # 是否双向LSTM
        

        self.num_epochs = 50                                           # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.learning_rate = 1e-4                                      # 学习率
        

        nowTime = get_now_time()
        self.model_name = 'output/LSTM_'+nowTime

class Model(nn.Module):

    def __init__(self,config):
        super(Model, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.hidden_layers = config.hidden_layers
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.bidirectional = config.BiDirection

        # 设置lstm网络结构参数
        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                            num_layers=self.hidden_layers,batch_first=True,
                            dropout=self.dropout,bidirectional = self.bidirectional)

        if self.bidirectional == False:
            self.fc = nn.Linear(self.hidden_layers * self.hidden_dim,self.num_classes)
        else:
            self.fc = nn.Linear(2 * self.hidden_layers * self.hidden_dim,self.num_classes)

    def forward(self, data):
        #print("data shape : ",data.size())
        # data shape : (batch_size,sequence_len,hidden_size)
        x,(h_n,c_n) = self.lstm(data)

        if self.bidirectional == True or self.hidden_layers > 1:
            h_n = torch.swapaxes(h_n,0,1)
            #print("h_n shape : ",h_n.shape)
            h_n_flatten = h_n.reshape(data.shape[0],-1)
        else:
            h_n_flatten = h_n

        out = self.fc(h_n_flatten)  # 获取最后一个h输出作为全连接层输入
    
        return out.squeeze()
        

if __name__ == '__main__':
    # 使用torchsummary打印模型信息

    config = Config()
    M = Model(config).to(config.device)

    from torchsummary import summary

    a = torch.randn(10,20,300)
    print(a.shape)
    d = M(a.to(config.device))
    print(d.shape)


    summary(M,(20,300))
