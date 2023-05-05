from torch import nn
import torch
import torch.nn.functional as F
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

        self.num_classes = 3                                        # 类别数

        self.used_embed = False                                     # 是否使用自己的词表做词嵌入  True -- 使用   False -- 使用现成的word2vec向量
        self.pad_size = 64          
        self.vocab_path = "data/vocab.pkl"                                        # 词表位置
        self.num_vocab = 20000                                      # 词表大小


        self.dropout = 0.2                                          # 随机失活
        
        self.input_dim = 300                                        # 输入向量维度
        self.hidden_size = 300                                      # 隐藏层特征维度
        self.hidden_layers = 3                                      # LSTM隐藏层个数

        self.num_epochs = 50                                           # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.learning_rate = 1e-4                                      # 学习率
        self.fc_linear_size = [2048,512]                               # 两层全连接层结构
        

        nowTime = get_now_time()
        self.model_name = 'output/ESIM_'+nowTime



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        #self.args = args
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.embeds_dim = config.input_dim
        num_word = config.num_vocab
        self.fc_linear_size = config.fc_linear_size
        self.num_classes = config.num_classes
        self.used_embed = config.used_embed

        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.fc_linear_size[0]),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.fc_linear_size[0]),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_linear_size[0], self.fc_linear_size[1]),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.fc_linear_size[1]),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_linear_size[1], self.num_classes),
            nn.Softmax(dim=-1)
        )
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # mask2等待匹配下面与x2相乘
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, input):

        if self.used_embed:
            # batch_size * seq_len
            sent1, sent2 = input[0], input[1]
            mask1, mask2 = sent1.eq(0), sent2.eq(0)
            #print('sent1 shape : ',sent1.shape)
            #print('mask1 shape : ',mask1.shape)

            # embeds: batch_size * seq_len => batch_size * seq_len * dim
            x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
            x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        else:
            # batch_size * seq_len
            x1, x2 = input[0], input[1]
            mask1, mask2 = input[2].eq(0), input[3].eq(0)


        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity



if __name__ == '__main__':
    # 使用torchsummary打印模型信息

    config = Config()
    M = Model(config).to(config.device)

    a = torch.randint(1,10,(10,64)).to(config.device)
    b = torch.randint(1,10,(10,64)).to(config.device)
    print(a)
    mask1 = torch.zeros(10,64).to(config.device)
    mask2 = torch.zeros(10,64).to(config.device)
    for i in range(10):
        for j in range(23):
            mask1[i,j] = 1
            mask2[i,j] = 1
    d = M([a,b,mask1,mask2]).to(config.device)
    print(d)


