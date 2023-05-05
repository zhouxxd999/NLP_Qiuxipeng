from torch import nn
import torch
import torch.nn.functional as F
import time

START_TAG = "<START>"
STOP_TAG = "<STOP>"


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

        self.num_classes = 9                                        # 类别数
        self.batchsize = 8
      
        self.tag_to_ix = {"O": 0, "I-PER": 1, "I-ORG": 2,"I-LOC": 3, "I-MISC": 4, "B-MISC": 5, "B-ORG": 6, "B-LOC": 7, START_TAG: 8, STOP_TAG: 9}

        self.vocab_path = "data/vocab.pkl"                          # 词表位置
        self.num_vocab = -1                                      # 词表大小


        self.dropout = 0.2                                          # 随机失活
        
        self.embedding_dim = 300
        self.hidden_dim = 300

        self.num_epochs = 3                                           # epoch数
        self.learning_rate = 1e-4                                      # 学习率
        
        nowTime = get_now_time()
        self.model_name = 'output/BiLSTM_CRF_'+nowTime




def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix.keys():
            idxs.append(to_ix[w])
        else:
            idxs.append(len(to_ix.keys()))
    #idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.device = config.device
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.num_vocab
        self.tag_to_ix = config.tag_to_ix
        self.tagset_size = len(self.tag_to_ix)

        self.word_embeds = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(config.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        self.hidden = (self.hidden[0].to(self.device),self.hidden[1].to(self.device))
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var.to(self.device) + self.transitions[next_tag].to(self.device)
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        #print(sentence.device)
        feats = self._get_lstm_features(sentence)
        #print(feats.device)
        forward_score = self._forward_alg(feats)
        #print(forward_score.device)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    



if __name__ == '__main__':
    # 使用torchsummary打印模型信息

    config = Config()
    print(config.tag_to_ix)
    M = Model(config).to(config.device)


    d = "the wall street journal reported today that apple corporation made money"
    l = "B I I I O O O B I O O"


    pred = M(torch.tensor([1,2,3,4,5,6,7],dtype=torch.long).to(config.device))
    print(pred)