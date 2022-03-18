
from pytorch_transformers import BertPreTrainedModel,BertConfig
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
from vilmodel import DicModel
import torch
import torch.nn as nn
import pdb
from ipdb import set_trace

class DicAddActionPreTrain(BertPreTrainedModel):
    def __init__(self,config):
        super(DicAddActionPreTrain, self).__init__(config)

        self.config = config
        self.bert = DicModel(config)

        self.action = NextActionPrediction(self.config.hidden_size, self.config.action_space)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlmhead = BertOnlyMLMHead(self.config)
        self.is_match = NextActionPrediction(self.config.hidden_size, 2)   
        self.order = NextOrderPrediction(self.config.hidden_size, 7) 
        self.is_class3 = NextActionPrediction(self.config.hidden_size, 3) 
        self.criterion_act = nn.CrossEntropyLoss(ignore_index=-1, size_average=False)
        self.init_weights()
        self.tie_weights()


    def tie_weights(self):
        self._tie_or_clone_weights(self.mlmhead.predictions.decoder,self.bert.embeddings.word_embeddings)


    def forward(self, seq, labels, ismatch=None, f_t_all = None,lang_mask=None, img_mask =None, task= None):

        ctx, pooled_out, attended_language, attended_visual, lang_pooler, visn_pooler = self.bert(seq, attention_mask=lang_mask,img_feats=f_t_all, img_mask=img_mask)
        if task == 'mlm':
            lang_part = ctx
            prediction_scores = self.mlmhead(lang_part)
            mask_loss = self.criterion(prediction_scores.view(-1,self.config.vocab_size), labels.view(-1))
            loss = mask_loss
            return mask_loss, prediction_scores          

        elif task == 'itm':
            cls_part = lang_pooler * visn_pooler
            match_scores = self.is_match(cls_part)
            match_loss = self.criterion(match_scores, ismatch) * 5
            return match_loss, match_scores

        elif task == 'order':
            visn_part = attended_visual
            match_scores = self.order(visn_part)
            match_loss = self.criterion(match_scores, ismatch) * 5
            return match_loss, match_scores    

        elif task == '3class':
            cls_part = ctx[:, 0, :]
            match_scores = self.is_class3(cls_part)
            match_loss = self.criterion(match_scores, ismatch) * 10 
            return match_loss, match_scores     

        else:
            cls_part = ctx[:, 0, :]
            match_scores = self.action(cls_part)
            match_loss = self.criterion(match_scores, ismatch)          
            return match_loss, match_scores   



class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))




class NextImgPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # the 0-35 is the vision, 36th is the CLS token


class NextActionPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # the 0-35 is the vision, 36th is the CLS token

class NextOrderPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # the 0-35 is the vision, 36th is the CLS token