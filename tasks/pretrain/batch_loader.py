import glob
import os, argparse, json
import time, copy, random, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, SplitTokenizer, padding_idx, \
    timeSince, boolean_string, preprocess_get_pano_states, current_best
from feature import Feature, Feature_bnb
from pytorch_transformers import BertTokenizer
import pprint
import pdb
from ipdb import set_trace



angle_inc = np.pi / 6.
feature_store = 'img_features/ResNet-152-imagenet.tsv'
panoramic = True

def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex,  0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex,   96:] = np.cos(rel_elevation)
    return embedding

# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]

def build_viewpoint_loc_embedding_1(viewIndex, absViewIndex):
    embedding = np.zeros((128), np.float32)
    relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
    rel_heading = (relViewIndex % 12) * angle_inc
    rel_elevation = (relViewIndex // 12 - 1) * angle_inc
    embedding[0:32] = np.sin(rel_heading)
    embedding[32:64] = np.cos(rel_heading)
    embedding[64:96] = np.sin(rel_elevation)
    embedding[96:] = np.cos(rel_elevation)
    return embedding

class SingleQuery(object):
    """
    A single data example for pre-training
    """
    def __init__(self, instr_id, scan, viewpoint, viewIndex, teacher_action, teacher_action_embedding, absViewIndex, rel_heading, rel_elevation, mode):
        self.instr_id = instr_id
        self.scan = scan
        self.viewpoint = viewpoint
        self.viewIndex = viewIndex
        self.teacher_action = teacher_action
        self.teacher_action_embedding = teacher_action_embedding
        self.absViewIndex = absViewIndex
        self.rel_heading = rel_heading
        self.rel_elevation = rel_elevation
        self.next = None
        self.mode = mode

class SingleQuery_bnb(object):
    """
    A single data example for pre-training
    """
    def __init__(self, instr_id, picid, mode):
        self.instr_id = instr_id
        self.picid = picid
        self.mode = mode


def new_mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.ByteTensor), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).type(torch.ByteTensor)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.ByteTensor) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.ByteTensor) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def mask_tokens_nomlm(inputs, tokenizer, args):

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    att_mask = [val == tokenizer.pad_token_id for val in labels.tolist()]

    attention_mask = torch.full(labels.shape, 1).masked_fill_(torch.tensor(att_mask, dtype=torch.uint8), value=0)

    return inputs, labels


def mask_tokens(inputs, tokenizer, args):

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    #probability_matrix = torch.full(labels.shape, args.mlm_probability)
    probability_matrix = torch.full(labels.shape, 0.15)    
    special_tokens_mask = [val in tokenizer.all_special_ids for val in labels.tolist()]
    att_mask = [val == tokenizer.pad_token_id for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.uint8), value=0.0)
    #masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).type(torch.ByteTensor)
    masked_indices = torch.bernoulli(probability_matrix).type(torch.ByteTensor)

    attention_mask = torch.full(labels.shape, 1).masked_fill_(torch.tensor(att_mask, dtype=torch.uint8), value=0)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens


    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.ByteTensor) & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)



    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.ByteTensor) & masked_indices & ~indices_replaced

    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, attention_mask

class NavDataset(data.Dataset):

    def __init__(self, json_dirs, bnb_dir, tok, img_path, panoramic, args, img_path_bnb):

        # read all json files and create a list of query data
        self.json_dirs = json_dirs  #  a list of json files
        self.tok = tok    # should be a lang, vision, action aware tokenizer ['VCLS', 'ACLS']
        self.mask_index = tok._convert_token_to_id(tok.mask_token)
        self.feature_store = Feature(img_path, panoramic)

        self.feature_store_bnb = Feature_bnb(img_path_bnb)
        self.args = args
        self.nag_dirs = 'data/neg/candidate_sm.json'
        self.nag_trajs = []
        self.nag_dirs_seq = 'data/neg/candidate_sm_seq.json'
        with open(self.nag_dirs_seq) as f:
            nag_trajs_seq = json.load(f)
        self.nag_trajs_seq = nag_trajs_seq
        self.nag_dirs_bnb = 'data/neg/bnb_neg.json'
        with open(self.nag_dirs_bnb) as f:
            nag_trajs_bnb = json.load(f)
        self.nag_trajs_bnb = nag_trajs_bnb
        self.data = []
        self.instr_refer_bnb = dict()  # instr_id : instr_encoding
        if bnb_dir is not None:
            with open(bnb_dir) as f:
                current_trajs = json.load(f)
                for traj in current_trajs:
                    self.data += self.disentangle_path_bnb(traj)

        self.instr_refer = dict()  # instr_id : instr_encoding
        if self.json_dirs is not None:
            for json_dir in self.json_dirs:
                with open(json_dir) as f:
                    current_trajs = json.load(f)
                    for traj in current_trajs:
                        self.data += self.disentangle_path(traj)


    def __getitem__(self, index):
        # you must return data and label pair tensor
        query = self.data[index]
        index_list = range(len(self.data))
        output = self.getQuery(query, self.nag_trajs, self.nag_trajs_seq, self.nag_trajs_bnb)
        return {key:torch.tensor(value, dtype = torch.float32) for key,value in output.items()}


    def __len__(self):
        return len(self.data)


    def disentangle_path(self, traj):
        mode = 'r2r'
        query = list()
        instr_id = traj['instr_id']
        instruction = traj['instr_encoding']
        self.instr_refer[instr_id] = instruction

        path = traj['path']
        actions = traj['teacher_actions']
        action_emds = traj['teacher_action_emd']

        scan, viewpoint, viewIndex, absViewIndex, rel_heading, rel_elevation, teacher_action, teacher_action_embedding =[],[],[],[],[],[],[],[]
        for t in range(len(path)):
            scan_now = path[t][0]
            viewpoint_now = path[t][1]
            viewIndex_now = path[t][2]
            teacher_action_now = actions[t]
            teacher_action_emd_now = action_emds[t]
            absViewIndex_now, rel_heading_now, rel_elevation_now = action_emds[t]

            scan.append(scan_now)
            viewpoint.append(viewpoint_now)
            viewIndex.append(viewIndex_now)
            absViewIndex.append(absViewIndex_now)
            rel_heading.append(rel_heading_now)
            rel_elevation.append(rel_elevation_now)
            teacher_action.append(teacher_action_now)
            teacher_action_embedding.append(teacher_action_emd_now)


        current_query = SingleQuery(instr_id, scan, viewpoint, viewIndex, teacher_action, teacher_action_embedding, absViewIndex, rel_heading, rel_elevation, mode)
        query.append(current_query)  # a list of (SASA)

        return query


    def disentangle_path_bnb(self, traj):
        mode = 'bnb'
        query_bnb = list()
        instr_id_bnb = traj['instr_id']
        instruction_bnb = traj['instr_encoding']
        self.instr_refer_bnb[instr_id_bnb] = instruction_bnb

        picid = traj['path']

        current_query_bnb = SingleQuery_bnb(instr_id_bnb, picid, mode)
        query_bnb.append(current_query_bnb)  # a list of (SASA)

        return query_bnb

    def getQuery(self, query, nag_trajs, nag_trajs_seq, nag_trajs_bnb):
        # prepare text tensor
        output = dict()
        if query.mode == 'bnb':
            text_seq = torch.LongTensor(self.instr_refer_bnb[query.instr_id])
            masked_text_seq, masked_text_label, attention_mask = mask_tokens(text_seq, self.tok, self.args)
            output['masked_text_seq'] = masked_text_seq
            output['masked_text_label'] = masked_text_label
            output['lang_attention_mask'] = attention_mask
            

            nomasked_text_seq, nomasked_text_label = mask_tokens_nomlm(text_seq, self.tok, self.args)
            output['text_seq'] = nomasked_text_seq
            output['text_label'] = nomasked_text_label

            # prepare vision tensor
            picid = query.picid
            rel_heading, rel_elevation = 0, 0

            feature_single_bnb = np.zeros((7,2176))
            img_mask_bnb = np.zeros(7)

            for t in range(len(picid)):
                feature_1_bnb = self.feature_store_bnb.rollout(picid[t])
                feature_angle_bnb = build_viewpoint_loc_embedding_1(0, 0)
                feature_single_bnb[t] = np.concatenate((feature_1_bnb, feature_angle_bnb), axis=-1)
                img_mask_bnb[t] = 1

            output['feature_single'] = feature_single_bnb #10, 2048
            output['img_mask'] = img_mask_bnb

            # prepare itm vision tensor
            prob_itm = np.random.random()
            prob_itm = 0.1
            if prob_itm <= 0.5:
                output['ismatch'] = 1
                output['feature_single_itm'] = output['feature_single']
                output['img_mask_itm'] = img_mask_bnb 
            elif prob_itm <= 1:
                output['ismatch'] = 0
                fake_feature_single_bnb = np.zeros((7,2176))
                fake_img_mask_bnb = np.zeros(7)
                #fake_feature_single_bnb = np.zeros((7,2176)) + feature_single_bnb
                can_room = list(range(len(nag_trajs_bnb)))
                fake_index = np.random.choice(can_room)
                fake_room = nag_trajs_bnb[fake_index]['instr_id'].replace("_0","")
                fake_picid = nag_trajs_bnb[fake_index]['path']
                true_room = query.instr_id.replace("_0","").replace("_1","").replace("_2","").replace("_3","").replace("_4","").replace("_5","").replace("_6","").replace("_7","").replace("_8","")
                while fake_room == true_room:
                    fake_index = np.random.choice(can_room)
                    fake_room = nag_trajs_bnb[fake_index]['instr_id'].replace("_0","")
                    fake_picid = nag_trajs_bnb[fake_index]['path']
                
                for t in range(len(fake_picid)):
                    fake_feature_1_bnb = self.feature_store_bnb.rollout(fake_picid[t])
                    fake_feature_angle_bnb = build_viewpoint_loc_embedding_1(0, 0)
                    fake_feature_single_bnb[t] = np.concatenate((fake_feature_1_bnb, fake_feature_angle_bnb), axis=-1)
                    fake_img_mask_bnb[t] = 1
                output['feature_single_itm'] = fake_feature_single_bnb #10, 2048
                output['img_mask_itm'] = fake_img_mask_bnb
            
            # prepare order vision random
            len_img = len(picid)
            prob_order = np.random.random()
            prob_order = 0.1
            if prob_order < 0.3:
                output['feature_single_order'] = feature_single_bnb
                if len_img == 5:
                    list_order_all = np.array([0, 1, 2, 3, 4, -1, -1])
                    list_order = list(range(len_img))
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                elif len_img == 6:
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, -1])
                    list_order = list(range(len_img))
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                else:
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, 6])
                    list_order = list(range(len_img))
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
            else:
                num_change = 3
                if len_img == 5 :
                    list_order_all = np.array([0, 1, 2, 3, 4, -1, -1])
                    list_order = list(range(len_img))
                    random_order = random.sample(list_order, num_change)
                    aaa = random_order[0]
                    bbb = random_order[1]
                    ccc = random_order[2]
                    list_order[aaa] = bbb
                    list_order[bbb] = ccc
                    list_order[ccc] = aaa
                    feature_init = np.zeros((7,2176))
                    shuffle_feature_single = feature_single_bnb[list_order]
                    feature_init[0:len_img] = shuffle_feature_single
                    output['feature_single_order'] = feature_init
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                elif len_img == 6 :
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, -1])
                    list_order = list(range(len_img))
                    random_order = random.sample(list_order, num_change)
                    aaa = random_order[0]
                    bbb = random_order[1]
                    ccc = random_order[2]
                    list_order[aaa] = bbb
                    list_order[bbb] = ccc
                    list_order[ccc] = aaa
                    feature_init = np.zeros((7,2176))
                    shuffle_feature_single = feature_single_bnb[list_order]
                    feature_init[0:len_img] = shuffle_feature_single
                    output['feature_single_order'] = feature_init
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                else :
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, 6])
                    list_order = list(range(len_img))
                    random_order = random.sample(list_order, num_change)
                    aaa = random_order[0]
                    bbb = random_order[1]
                    ccc = random_order[2]
                    list_order[aaa] = bbb
                    list_order[bbb] = ccc
                    list_order[ccc] = aaa
                    feature_init = np.zeros((7,2176))
                    shuffle_feature_single = feature_single_bnb[list_order]
                    feature_init[0:len_img] = shuffle_feature_single
                    output['feature_single_order'] = feature_init
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all

            # prepare 3-class vision random
            prob_class3 = np.random.random()
            prob_class3 = 0.1
  
            if prob_class3 <= 0.34:
                output['isclass'] = 1
                output['feature_single_class'] = output['feature_single']
            elif prob_class3 <= 0.67:
                output['isclass'] = 2
                fake_feature_single_class = np.zeros((7,2176))
                if len(picid) == 5:
                    list_order_all = torch.LongTensor([3,4,0,1,2,5,6])
                elif len(picid) == 6:
                    list_order_all = torch.LongTensor([3,4,5,0,1,2,6])   
                else:
                    list_order_all = torch.LongTensor([3,4,5,6,0,1,2])                         
                fake_feature_single_class = feature_single_bnb[list_order_all]
                output['feature_single_class'] = fake_feature_single_class
            else:
                output['isclass'] = 0
                fake_feature_single_class = np.zeros((7,2176))
                can_room = list(range(len(nag_trajs_bnb)))
                fake_index = np.random.choice(can_room)
                fake_room = nag_trajs_bnb[fake_index]['instr_id'].replace("_0","")
                fake_picid = nag_trajs_bnb[fake_index]['path']
                true_room = query.instr_id.replace("_0","").replace("_1","").replace("_2","").replace("_3","").replace("_4","").replace("_5","").replace("_6","").replace("_7","").replace("_8","")
                while fake_room == true_room:
                    fake_index = np.random.choice(can_room)
                    fake_room = nag_trajs_bnb[fake_index]['instr_id'].replace("_0","")
                    fake_picid = nag_trajs_bnb[fake_index]['path']
  
                for t in range(3):
                    fake_feature_1_class = self.feature_store_bnb.rollout(fake_picid[t])
                    fake_feature_angle_class = build_viewpoint_loc_embedding_1(0, 0)
                    fake_feature_single_class[t] = np.concatenate((fake_feature_1_class, fake_feature_angle_class), axis=-1)
                      
                output['feature_single_class'] = fake_feature_single_class

            #action
            feature_history = np.zeros((7,2176))
            feature_36 = np.zeros((36,2048))
            feature_36_with_loc_all = np.concatenate((feature_36, _static_loc_embeddings[0]), axis=-1)
            img_mask_his = np.zeros(7)

            output['teacher_embedding'] = -1
            output['feature_his'] = feature_history
            output['feature_36'] = feature_36_with_loc_all
            output['img_mask_his_36'] = np.concatenate((img_mask_his, np.zeros(1), np.zeros(36)+1), axis=-1)

        elif query.mode == 'r2r':
            text_seq = torch.LongTensor(self.instr_refer[query.instr_id])
            masked_text_seq, masked_text_label, attention_mask = mask_tokens(text_seq, self.tok, self.args)
            output['masked_text_seq'] = masked_text_seq
            output['masked_text_label'] = masked_text_label
            output['lang_attention_mask'] = attention_mask

            nomasked_text_seq, nomasked_text_label = mask_tokens_nomlm(text_seq, self.tok, self.args)
            output['text_seq'] = nomasked_text_seq
            output['text_label'] = nomasked_text_label

            # prepare vision tensor
            scan, viewpoint, viewindex = query.scan, query.viewpoint, query.viewIndex
            absViewIndex, rel_heading, rel_elevation = query.absViewIndex, query.rel_heading, query.rel_elevation

            feature_single = np.zeros((7,2176))
            img_mask = np.zeros(7)

            for t in range(len(scan)):
                feature_1, feature_all = self.feature_store.rollout(scan[t], viewpoint[t], viewindex[t])
                feature_angle = build_viewpoint_loc_embedding_1(viewindex[t], absViewIndex[t])
                feature_single[t] = np.concatenate((feature_1, feature_angle), axis=-1)
                img_mask[t] = 1
            output['feature_single'] = feature_single 
            output['img_mask'] = img_mask

            # prepare itm vision tensor
            prob_itm = np.random.random()
            if prob_itm <= 0.5:
                output['ismatch'] = 1
                output['feature_single_itm'] = output['feature_single']
                output['img_mask_itm'] = img_mask 
            elif prob_itm <= 1:
                output['ismatch'] = 0
                fake_feature_single_itm = np.zeros((7,2176))
                can_scan = list(range(len(nag_trajs_seq)))
                fake_index = np.random.choice(can_scan)
                fake_scan = nag_trajs_seq[fake_index]['scan']

                while fake_scan == scan[0]:
                    fake_index = np.random.choice(can_scan)
                    fake_scan = nag_trajs_seq[fake_index]['scan'] 
                    if len(nag_trajs_seq[fake_index]['path']) == 0:
                        fake_index = np.random.choice(can_scan)
                        fake_scan = nag_trajs_seq[fake_index]['scan'] 

                list_path = range(len(nag_trajs_seq[fake_index]['path']))
                random_path = random.sample(list_path, 1)
                fake_path = nag_trajs_seq[fake_index]['path'][random_path[0]]

                fake_img_mask = np.zeros(7)
                for t in range(len(fake_path)):
                    if t == len(fake_path) - 1:
                        fake_absviewindex_itm = -1
                    else:
                        fake_absviewindex_itm = fake_path[t+1][2] 

                    fake_viewpoint_itm = fake_path[t][1]
                    fake_viewindex_itm = fake_path[t][2]
                    fake_feature_1_itm, fake_feature_1_itm_all = self.feature_store.rollout(fake_scan, fake_viewpoint_itm, fake_viewindex_itm)
                    feature_angle_itm = build_viewpoint_loc_embedding_1(fake_viewindex_itm, fake_absviewindex_itm)                
                    fake_feature_single_itm[t] = np.concatenate((fake_feature_1_itm, feature_angle_itm), axis=-1)
                    fake_img_mask[t] = 1
                    output['feature_single_itm'] = fake_feature_single_itm
                    output['img_mask_itm'] = fake_img_mask  

            # prepare order vision random
            len_img = len(scan)
            prob_order = np.random.random()

            if prob_order < 0.3:
                output['feature_single_order'] = feature_single
                if len_img == 5:
                    list_order_all = np.array([0, 1, 2, 3, 4, -1, -1])
                    list_order = list(range(len_img))
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                elif len_img == 6:
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, -1])
                    list_order = list(range(len_img))
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                else:
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, 6])
                    list_order = list(range(len_img))
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
            else:
                num_change = 3
                if len_img == 5 :
                    list_order_all = np.array([0, 1, 2, 3, 4, -1, -1])
                    list_order = list(range(len_img))
                    random_order = random.sample(list_order, num_change)
                    aaa = random_order[0]
                    bbb = random_order[1]
                    ccc = random_order[2]
                    list_order[aaa] = bbb
                    list_order[bbb] = ccc
                    list_order[ccc] = aaa
                    feature_init = np.zeros((7,2176))
                    shuffle_feature_single = feature_single[list_order]
                    feature_init[0:len_img] = shuffle_feature_single
                    output['feature_single_order'] = feature_init
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                elif len_img == 6 :
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, -1])
                    list_order = list(range(len_img))
                    random_order = random.sample(list_order, num_change)
                    aaa = random_order[0]
                    bbb = random_order[1]
                    ccc = random_order[2]
                    list_order[aaa] = bbb
                    list_order[bbb] = ccc
                    list_order[ccc] = aaa
                    feature_init = np.zeros((7,2176))
                    shuffle_feature_single = feature_single[list_order]
                    feature_init[0:len_img] = shuffle_feature_single
                    output['feature_single_order'] = feature_init
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all
                else :
                    list_order_all = np.array([0, 1, 2, 3, 4, 5, 6])
                    list_order = list(range(len_img))
                    random_order = random.sample(list_order, num_change)
                    aaa = random_order[0]
                    bbb = random_order[1]
                    ccc = random_order[2]
                    list_order[aaa] = bbb
                    list_order[bbb] = ccc
                    list_order[ccc] = aaa
                    feature_init = np.zeros((7,2176))
                    shuffle_feature_single = feature_single[list_order]
                    feature_init[0:len_img] = shuffle_feature_single
                    output['feature_single_order'] = feature_init
                    list_order_all[0:len_img] = list_order
                    list_order_all = torch.LongTensor(list_order_all)
                    output['order'] = list_order_all

            # prepare 3-class vision random
            prob_class3 = np.random.random()

            if prob_class3 <= 0.34:
                output['isclass'] = 1
                output['feature_single_class'] = output['feature_single']
            elif prob_class3 <= 0.67:
                output['isclass'] = 2
                fake_feature_single_class = np.zeros((7,2176)) + feature_single
                if len(scan) == 5:
                    list_order_all = torch.LongTensor([3,4,0,1,2,5,6])
                elif len(scan) == 6:
                    list_order_all = torch.LongTensor([3,4,5,0,1,2,6])   
                else:
                    list_order_all = torch.LongTensor([3,4,5,6,0,1,2])                        
                fake_feature_single_class = feature_single[list_order_all]
                output['feature_single_class'] = fake_feature_single_class
            else:
                output['isclass'] = 0
                fake_feature_single_class = np.zeros((7,2176)) + feature_single
                can_scan = list(range(len(nag_trajs_seq)))
                fake_index = np.random.choice(can_scan)
                fake_scan = nag_trajs_seq[fake_index]['scan']

                while fake_scan == scan[0]:
                    fake_index = np.random.choice(can_scan)
                    fake_scan = nag_trajs_seq[fake_index]['scan'] 
                    if len(nag_trajs_seq[fake_index]['path']) == 0:
                        fake_index = np.random.choice(can_scan)
                        fake_scan = nag_trajs_seq[fake_index]['scan'] 

                list_path = range(len(nag_trajs_seq[fake_index]['path']))
                random_path = random.sample(list_path, 1)
                fake_path = nag_trajs_seq[fake_index]['path'][random_path[0]]

                for t in range(3):
                    fake_viewpoint = fake_path[t][1]
                    fake_viewindex = fake_path[t][2]
                    fake_absviewindex = fake_path[t+1][2]             
                    fake_feature_1, fake_feature_1_all = self.feature_store.rollout(fake_scan, fake_viewpoint, fake_viewindex)
                    fake_feature_angle = build_viewpoint_loc_embedding_1(fake_viewindex, fake_absviewindex)
                    fake_feature_single_class[t] = np.concatenate((fake_feature_1, fake_feature_angle), axis=-1)
                            
                output['feature_single_class'] = fake_feature_single_class

            # prepare action
            select_step = np.random.choice(len(scan)-2)
            if select_step == 0:
        	    select_step = np.random.choice(len(scan)-2)

            feature_history = np.zeros((7,2176))
            img_mask_his = np.zeros(7)
        
            if select_step > 0:
        	    for t in range(select_step):
            		feature_history_1, feature_all = self.feature_store.rollout(scan[t], viewpoint[t], viewindex[t])
            		feature_history_angle = build_viewpoint_loc_embedding_1(viewindex[t], absViewIndex[t])
            		feature_history[t] = np.concatenate((feature_history_1, feature_angle), axis=-1)
        	    	img_mask_his[t] = 1
            feature_36_1, feature_36 = self.feature_store.rollout(scan[select_step + 1], viewpoint[select_step + 1], viewindex[select_step + 1])
            feature_36_with_loc_all = np.concatenate((feature_36, _static_loc_embeddings[viewindex[select_step + 1]]), axis=-1)
            output['teacher_embedding'] = query.teacher_action_embedding[select_step+1][0]
            output['feature_his'] = feature_history
            output['feature_36'] = feature_36_with_loc_all

            output['img_mask_his_36'] = np.concatenate((img_mask_his, np.zeros(1), np.zeros(36)+1), axis=-1)


        return output




    def random_word(self, text_seq):
        tokens = text_seq.copy()   # already be [cls t1 t2 sep]
        output_label = []

        for i, token in enumerate(tokens):
            if i ==0 or i == len(tokens) - 1:
                output_label.append(0)
                continue
            prob = np.random.random()
            if prob < 0.15:
                prob /= 0.15

                output_label.append(tokens[i])

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.tok))

                # 10% randomly change token to current token
                else:
                    tokens[i] = tokens[i]   # just keep it

            else:
                tokens[i] = tokens[i]   # just keep it
                output_label.append(0)

        return tokens, output_label









