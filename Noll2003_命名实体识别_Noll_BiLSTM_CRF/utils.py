import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
from tqdm import tqdm
import os
import pickle as pkl
import time


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def build_vocab(dataset, tokenizer, max_size, min_freq):
    vocab_dic = {}
    
    data = [' '.join(d[0]) for d in dataset]

    for i in range(len(data)):
        content = data[i]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic



def build_dataset(config,dataset,use_word=True,isTest=False):
    
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    if os.path.exists(config.vocab_path):
        print('Loading exist vocab ... ')
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        print('Creating new vocab ... ')
        vocab = build_vocab(dataset, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(datas, isTest=False):

        tag_to_ix = {"O": 0, "I-PER": 1, "I-ORG": 2,"I-LOC": 3, "I-MISC": 4, "B-MISC": 5, "B-ORG": 6, "B-LOC": 7, START_TAG: 8, STOP_TAG: 9}
    
        data = []
        label = []
        for d in datas:
            da = d[0]
            la = d[1]
            if isTest == False:
                label.append([tag_to_ix[l] for l in la])
            else:
                label.append([0] * len(la))
            
            word2idx = []
            for word in da:
                word2idx.append(vocab.get(word, vocab.get(UNK)))
            data.append(word2idx)
    
    

        return data,label


    # 创建数据集
    data,label = load_dataset(datas = dataset,isTest=isTest)

    return vocab,data,label


# 定义数据读取类
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self,data,label):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
 
        self.data = data
        self.label = label


    def __getitem__(self, index):
        """
        步骤三:实现__getitem__方法,定义指定index时如何获取数据,并返回单条数据(训练数据，对应的标签)
        """
        d = torch.as_tensor(self.data[index],dtype=torch.long)   
        l = torch.as_tensor(self.label[index],dtype=torch.long)

        return d,l

    def __len__(self):
        """
        步骤四:实现__len__方法:返回数据集总数目
        """
        return len(self.label)







if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./huaxue/data/train.txt"
    vocab_dir = "./huaxue/data/vocab.pkl"
    pretrain_dir = "./huaxue/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./huaxue/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
        print(word_to_id)
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='gbk')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)




