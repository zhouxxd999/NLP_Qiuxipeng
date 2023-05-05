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




def build_vocab(df, tokenizer, max_size, min_freq):
    vocab_dic = {}
    
    df_sen1 = df['sentence1'].values.tolist()
    df_sen2 = df['sentence2'].values.tolist()

    for i in range(len(df)):
        content = df_sen1[i] + ' ' + df_sen2[i]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic



def build_dataset(config,df,use_word=True,isTest=False):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    if os.path.exists(config.vocab_path):
        print('Loading exist vocab ... ')
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        print('Creating new vocab ... ')
        vocab = build_vocab(df, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(df, pad_size=64,isTest=False):
        contents = []

        df_sen1 = df['sentence1'].values.tolist()
        df_sen2 = df['sentence2'].values.tolist()

        if isTest == False:
            label = df['gold_label'].values.tolist()

        for i in range(len(df)):
            words_line1 = []
            words_mask1 = []
            token = tokenizer(df_sen1[i])
            if pad_size:
                if len(token) < pad_size:    # 句子长度短了，在末尾添加[PAD]
                    token.extend([PAD] * (pad_size - len(token)))
                else:                        # 句子长了，就切除末尾字符串
                    token = token[:pad_size]               
    
            # 将句子根据词表转换成ID，以便于后面word embedding
            for word in token:
                words_line1.append(vocab.get(word, vocab.get(UNK)))
                if word == PAD:
                    words_mask1.append(0)
                else:
                    words_mask1.append(1)

            words_line2 = []
            words_mask2 =  []
            token = tokenizer(df_sen2[i])
            if pad_size:
                if len(token) < pad_size:    # 句子长度短了，在末尾添加[PAD]
                    token.extend([PAD] * (pad_size - len(token)))
                else:                        # 句子长了，就切除末尾字符串
                    token = token[:pad_size]
    
            # 将句子根据词表转换成ID，以便于后面word embedding
            for word in token:
                words_line2.append(vocab.get(word, vocab.get(UNK)))
                if word == PAD:
                    words_mask2.append(0)
                else:
                    words_mask2.append(1)
            
            if isTest == False:
                
                l = int(['entailment','neutral','contradiction'].index(label[i]))
                contents.append(((words_line1,words_line2,words_mask1,words_mask2),int(l)))
            else:
                contents.append(((words_line1,words_line2,words_mask1,words_mask2),int(0)))

        return contents  # [(sentence1_word2id , sentence)]


    # 创建数据集
    df_n = load_dataset(df=df, pad_size= config.pad_size,isTest=isTest)

    return vocab,df_n


# 定义数据读取类
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self,dataset):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
 
        self.dataset = dataset


    def __getitem__(self, index):
        """
        步骤三:实现__getitem__方法,定义指定index时如何获取数据,并返回单条数据(训练数据，对应的标签)
        """
        d1 = torch.as_tensor(self.dataset[index][0][0],dtype=torch.int32)   
        d2 = torch.as_tensor(self.dataset[index][0][1],dtype=torch.long)
        mask1 = torch.as_tensor(self.dataset[index][0][2],dtype=torch.int32)   
        mask2 = torch.as_tensor(self.dataset[index][0][3],dtype=torch.float32)   
        la = torch.as_tensor(self.dataset[index][1],dtype=torch.long)

        return (d1,d2,mask1,mask2),la

    def __len__(self):
        """
        步骤四:实现__len__方法:返回数据集总数目
        """
        return len(self.dataset)







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




