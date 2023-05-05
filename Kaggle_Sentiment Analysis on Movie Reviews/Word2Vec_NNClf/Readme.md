# kaggle电影评论文本分类竞赛5分类
\
比赛地址 ： https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews  \
\
\
使用神经网络进行训练模型，进而分类\
\
可运行文件 : Word2Vec_NN.ipynb   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Word2Vec_NN_dataaug.ipynb &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Word2Vec_词频统计_加权重_NN.ipynb &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Word2Vec_词频统计_加特征_NN.ipynb
\
kaggle上面的分数都在**0.62**左右\
\
对数据进行预处理，使用空格分词，之后使用glove词表将句子转换成一维向量\
\
**Word2Vec_NN.ipynb** :  将向量直接输入全由线性层组成的神经网络，输出五分类\
\
**Word2Vec_NN_dataaug.ipynb** : 由于训练数据五分类数据不平衡，所以采用近义词替换的方式对训练数据进行数据增强\
\
**Word2Vec_词频统计_加权重_NN.ipynb**  : 观察到一些显著性词语（如'worse'大多出现在负面评价中）,将有显著性词语的词向量权重加倍\
\
**Word2Vec_词频统计_加特征_NN.ipynb**  ：将显著性词语做成一个one-hot向量，检测句子中是否出现对应的关键词并更新该向量，以该向量作为特征连接到词向量后面\
\
\
Word2Vec词表文件**glove.6B.300d.txt** 文件下载地址： https://nlp.stanford.edu/projects/glove/   
