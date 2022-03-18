from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pickle
import random
import pdb
import time
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from feature import Feature
from batch_loader import NavDataset
from tqdm import tqdm, trange
from pretrain_class import DicAddActionPreTrain
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer)
from ipdb import set_trace


feature_store = 'img_features/ResNet-152-imagenet.tsv'
feature_store_bnb = 'img_features/bnbdata.npz'
panoramic = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.getcwd() + '/logs/'
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG) 
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
}

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
                self.examples.append(tokenizer.add_special_tokens_single_sequence(tokenized_text[i:i+block_size]))

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset

def set_seed(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['n_gpu'] > 0:
        torch.cuda.manual_seed_all(args['seed'])

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args['mlm_probability'])).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def trainval(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args['train_batch_size'] = args['per_gpu_train_batch_size'] * max(1, args['n_gpu'])
    train_sampler = RandomSampler(train_dataset) if args['local_rank'] == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'], num_workers = 8)
    if args['max_steps'] > 0:
        t_total = args['max_steps']
        args['num_train_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1
    else: #-1
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    data_length = len(train_dataloader)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    if args['fp16']: #False
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args['local_rank'] != -1: #-1
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank']],
                                                          output_device=args['local_rank'],
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", args['per_gpu_train_batch_size'])
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args['train_batch_size'] * args['gradient_accumulation_steps'] * (torch.distributed.get_world_size() if args['local_rank'] != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch", disable=True)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    total_acc_mlm = 0.0
    avg_acc_mlm = 0.0
    total_acc_itm = 0.0
    avg_acc_itm= 0.0
    total_acc_order = 0.0
    avg_acc_order = 0.0
    total_acc_class3 = 0.0
    avg_acc_class3 = 0.0
    total_acc_action = 0.0
    avg_acc_action = 0.0
    step_mlm = 1
    step_itm = 1
    step_order = 1
    step_class3 = 1 
    step_action = 1 
    for epo in train_iterator:
        epoch_iterator =  tqdm(enumerate(train_dataloader),
                              desc="Iteration" ,
                              total=len(train_dataloader),
                              bar_format="{l_bar}{r_bar}",
                              disable=False)
        if epo < 3:
            tasks = ['mlm', 'itm', 'order', '3class', 'action']
        else:
            tasks = ['mlm', 'itm', 'order', '3class', 'action']
        for step, batch in epoch_iterator:
            task = random.choice(tasks)
            #task = '3class'
            inputs_mlm, labels_mlm = batch['masked_text_seq'].long(), batch['masked_text_label'].long()
            lang_attention_mask = batch['lang_attention_mask'].long() #bs 80

            img_feats = batch['feature_single'] #bs 36 2176
            img_feats = img_feats
            #actions = batch['teacher']
            img_mask = batch['img_mask']
            img_mask = img_mask

            inputs_mlm = inputs_mlm.to(args['device'])
            labels_mlm = labels_mlm.to(args['device'])
            img_mask = img_mask.to(args['device'])
            img_feats = img_feats.to(args['device'])
            lang_attention_mask = lang_attention_mask.to(args['device'])

            #itm
            inputs, labels = batch['text_seq'].long(), batch['text_label'].long()
            img_feats_itm = batch['feature_single_itm']
            img_feats_itm = img_feats_itm 
            ismatch = batch['ismatch'].long() 
            img_mask_itm = batch['img_mask_itm']
            img_mask_itm = img_mask_itm

            inputs = inputs.to(args['device'])
            labels = labels.to(args['device'])
            img_feats_itm = img_feats_itm.to(args['device']) 
            ismatch = ismatch.to(args['device'])  
            img_mask_itm = img_mask_itm.to(args['device'])

            #order
            img_feats_order = batch['feature_single_order']
            img_feats_order = img_feats_order   
            order = batch['order'].long()

            img_feats_order = img_feats_order.to(args['device'])
            order = order.to(args['device'])

            #3class
            img_feats_class = batch['feature_single_class']
            img_feats_class = img_feats_class  
            isclass = batch['isclass'].long()  

            img_feats_class = img_feats_class.to(args['device'])
            isclass = isclass.to(args['device'])
            
            #action
            action = batch['teacher_embedding'].long()
            img_history = batch['feature_his']
            img_36 = batch['feature_36']
            img_sep = np.zeros((len(img_36), 1, 2176)) + 102
            img_his_36 = np.concatenate((img_history, img_sep, img_36), axis=1)
            img_his_36 = torch.from_numpy(img_his_36).float()
            img_mask_his_36 = batch['img_mask_his_36']

            action = action.to(args['device'])
            img_his_36 = img_his_36.to(args['device'])
            img_mask_his_36 = img_mask_his_36.to(args['device'])

            model.train()

            if task == 'mlm':
                outputs = model(inputs_mlm,labels_mlm,None,img_feats,lang_mask=lang_attention_mask,img_mask = img_mask, task= task)
            elif task == 'itm':
                outputs = model(inputs,labels,ismatch,img_feats_itm, lang_mask=lang_attention_mask, img_mask = img_mask_itm, task = task)
            elif task == 'order':
                outputs = model(inputs,labels,order,img_feats_order,lang_mask=lang_attention_mask, img_mask = img_mask, task = task)
            elif task == '3class':
                outputs = model(inputs,labels,isclass,img_feats_class,lang_mask=lang_attention_mask, img_mask = img_mask, task= task)
            else:
                outputs = model(inputs,labels,action,img_his_36,lang_mask=lang_attention_mask, img_mask = img_mask_his_36, task= task)                
            
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            prediction_score = outputs[1]

            if task == 'mlm':
                bool_label = labels_mlm > 0
                pred = prediction_score[bool_label, :].argmax(1)
                valid_labels = labels_mlm[bool_label]   
                acc_mlm = (pred == valid_labels).type(torch.float).mean() * 100.
                total_acc_mlm += acc_mlm
                step_mlm = step_mlm + 1
                avg_acc_mlm = total_acc_mlm /step_mlm
            elif task == 'itm':
                correct = prediction_score.argmax(dim=-1).eq(ismatch.cuda()).sum().item()
                acc_itm = correct / ismatch.nelement() *100
                total_acc_itm += acc_itm
                step_itm = step_itm + 1
                avg_acc_itm = total_acc_itm /(step_itm+1)
            elif task == 'order':
                correct = prediction_score.argmax(dim=-1).eq(order.cuda()).sum().item()
                acc_order = correct / order.nelement() *100
                total_acc_order += acc_order
                step_order = step_order +1
                avg_acc_order = total_acc_order / step_order 
            elif task == 'action':
                correct = prediction_score.argmax(dim=-1).eq(action.cuda()).sum().item()
                acc_action = correct / action.nelement() *100
                total_acc_action += acc_action
                step_action = step_action +1
                avg_acc_action = total_acc_action / step_action                 
            else:
                correct = prediction_score.argmax(dim=-1).eq(isclass.cuda()).sum().item()
                acc_class3 = correct / isclass.nelement() *100
                total_acc_class3 += acc_class3
                step_class3 = step_class3 + 1
                avg_acc_class3 = total_acc_class3 / step_class3    

            if args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                if args['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['local_rank'] in [-1, 0] and args['save_steps'] > 0 and global_step % data_length == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(epo))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if step % 200 == 0:
                print("\n")
                print("PROGRESS: {}%".format(round((epo * len(train_dataloader) + step) * 100 / t_total, 4)))
                print("EVALERR: {:.4f}%,avg_acc_mlm:{:.2f},avg_acc_itm:{:.2f},avg_acc_order:{:.2f},avg_acc_class:{:.2f}, avg_acc_action:{:.2f}".format(tr_loss / (global_step), avg_acc_mlm, avg_acc_itm, avg_acc_order, avg_acc_class3, avg_acc_action))
                logger.info("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, avg_acc_mlm:%.2f, avg_acc_itm:%.2f, avg_acc_order:%.2f, avg_acc_class:%.2f, avg_acc_class:%.2f " % (epo + 1, len(train_iterator), step + 1, len(train_dataloader), tr_loss / (global_step), avg_acc_mlm, avg_acc_itm, avg_acc_order, avg_acc_class3, avg_acc_action))
           
            if args['max_steps'] > 0 and global_step > args['max_steps']: #-1
                epoch_iterator.close()
                break

        if args['max_steps'] > 0 and global_step > args['max_steps']:
            train_iterator.close()
            break

        results = evaluate(args, eval_dataset, model, tokenizer)  

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']
    if not os.path.exists(eval_output_dir) and args['local_rank'] in [-1, 0]:
        os.makedirs(eval_output_dir)
    args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args['local_rank'] == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'], num_workers = 8)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    eval_loss_mlm = 0.0    
    eval_loss_itm = 0.0 
    eval_loss_order = 0.0    
    eval_loss_class = 0.0
    eval_loss_action = 0.0      
    nb_eval_steps = 1
    nb_eval_steps_mlm = 1   

    model.eval()

    total_acc_mlm = 0.0
    avg_acc_mlm = 0.0
    total_acc_itm = 0.0
    avg_acc_itm= 0.0
    total_acc_order = 0.0
    avg_acc_order = 0.0
    total_acc_class = 0.0
    avg_acc_class = 0.0
    total_acc_action = 0.0
    avg_acc_action = 0.0
    step_mlm = 1
    step_itm = 1
    step_order = 1
    step_class3 = 1 
    step_action = 1
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs_mlm, labels_mlm = batch['masked_text_seq'].long(), batch['masked_text_label'].long()
        lang_attention_mask = batch['lang_attention_mask'].long() 

        img_feats = batch['feature_single'] 
        img_feats = img_feats
        img_mask = batch['img_mask']
        img_mask = img_mask

        inputs_mlm = inputs_mlm.to(args['device'])
        labels_mlm = labels_mlm.to(args['device'])
        img_mask = img_mask.to(args['device'])
        img_feats = img_feats.to(args['device'])
        lang_attention_mask = lang_attention_mask.to(args['device'])

        #itm
        inputs, labels = batch['text_seq'].long(), batch['text_label'].long()
        img_feats_itm = batch['feature_single_itm']
        img_feats_itm = img_feats_itm
        ismatch = batch['ismatch'].long() 
        img_mask_itm = batch['img_mask_itm']
        img_mask_itm = img_mask_itm

        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        img_feats_itm = img_feats_itm.to(args['device']) 
        ismatch = ismatch.to(args['device'])  
        img_mask_itm = img_mask_itm.to(args['device'])

        #order
        img_feats_order = batch['feature_single_order']
        img_feats_order = img_feats_order
        order = batch['order'].long()

        img_feats_order = img_feats_order.to(args['device'])
        order = order.to(args['device'])

        #3class
        img_feats_class = batch['feature_single_class']
        img_feats_class = img_feats_class  
        isclass = batch['isclass'].long()  

        img_feats_class = img_feats_class.to(args['device'])
        isclass = isclass.to(args['device'])

        #action
        action = batch['teacher_embedding'].long()
        img_history = batch['feature_his']
        img_36 = batch['feature_36']
        img_sep = np.zeros((len(img_36), 1, 2176)) + 102
        img_his_36 = np.concatenate((img_history, img_sep, img_36), axis=1)
        img_his_36 = torch.from_numpy(img_his_36).float()
        img_mask_his_36 = batch['img_mask_his_36']

        action = action.to(args['device'])
        img_his_36 = img_his_36.to(args['device'])
        img_mask_his_36 = img_mask_his_36.to(args['device'])

        tasklist = ['mlm', 'itm', 'order', '3class', 'action']

        with torch.no_grad():
            for task in tasklist:
                if task == 'mlm':
                    outputs = model(inputs_mlm,labels_mlm,None,img_feats,lang_mask=lang_attention_mask,img_mask = img_mask, task= task)
                    lm_loss_mlm = outputs[0]
                    eval_loss_mlm += lm_loss_mlm.mean().item()
                    prediction_score = outputs[1]
                    bool_label = labels_mlm > 0
                    pred = prediction_score[bool_label, :].argmax(1)
                    valid_labels = labels_mlm[bool_label]   
                    acc_mlm = (pred == valid_labels).type(torch.float).mean() * 100.
                    total_acc_mlm += acc_mlm
                    avg_acc_mlm = total_acc_mlm / nb_eval_steps
                elif task == 'itm':
                    outputs = model(inputs,labels,ismatch,img_feats_itm, lang_mask=lang_attention_mask, img_mask = img_mask_itm, task = task)
                    lm_loss_itm = outputs[0]
                    eval_loss_itm += lm_loss_itm.mean().item()
                    prediction_score = outputs[1]
                    correct = prediction_score.argmax(dim=-1).eq(ismatch.cuda()).sum().item()
                    acc_itm = correct / ismatch.nelement() *100
                    total_acc_itm += acc_itm
                    avg_acc_itm = total_acc_itm / nb_eval_steps
                elif task == 'order':
                    outputs = model(inputs,labels,order,img_feats_order,lang_mask=lang_attention_mask, img_mask = img_mask, task = task)
                    lm_loss_order = outputs[0]
                    eval_loss_order += lm_loss_order.mean().item()
                    prediction_score = outputs[1]
                    correct = prediction_score.argmax(dim=-1).eq(order.cuda()).sum().item()
                    acc_order = correct / order.nelement() *100
                    total_acc_order += acc_order
                    avg_acc_order = total_acc_order / nb_eval_steps
                elif task == 'action':
                    outputs = model(inputs,labels,action,img_his_36,lang_mask=lang_attention_mask, img_mask = img_mask_his_36, task= task)
                    lm_loss_action = outputs[0]
                    eval_loss_action += lm_loss_action.mean().item()
                    prediction_score = outputs[1]
                    correct = prediction_score.argmax(dim=-1).eq(action.cuda()).sum().item()
                    acc_action = correct / action.nelement() *100
                    total_acc_action += acc_action
                    avg_acc_action = total_acc_action / nb_eval_steps                    
                else:
                    outputs = model(inputs,labels,isclass,img_feats_class,lang_mask=lang_attention_mask, img_mask = img_mask, task= task)   
                    lm_loss_class = outputs[0]
                    eval_loss_class += lm_loss_class.mean().item()
                    prediction_score = outputs[1]
                    correct = prediction_score.argmax(dim=-1).eq(isclass.cuda()).sum().item()
                    acc_class = correct / isclass.nelement() *100
                    total_acc_class += acc_class
                    avg_acc_class = total_acc_class / nb_eval_steps

        nb_eval_steps += 1

        
    print("nb_eval_steps:", nb_eval_steps)

    eval_loss_mlm = eval_loss_mlm / nb_eval_steps
    eval_loss_itm = eval_loss_itm / nb_eval_steps
    eval_loss_order = eval_loss_order / nb_eval_steps
    eval_loss_class = eval_loss_class / nb_eval_steps
    eval_loss_action = eval_loss_action / nb_eval_steps

    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "loss_mlm": eval_loss_mlm,
        "loss_itm": eval_loss_itm,
        "loss_order": eval_loss_order,
        "loss_class": eval_loss_class,
        "loss_action": eval_loss_action,        
        "mlm_acc": avg_acc_mlm,
        "itm_acc": avg_acc_itm,
        "order_acc": avg_acc_order,
        "class_acc": avg_acc_class,
        "action_acc": avg_acc_action
    }

    logger.info("loss_mlm: %.4f, loss_itm: %.4f, loss_order: %.4f, loss_class: %.4f, acc_mlm:%.2f, acc_itm:%.2f, acc_order:%.2f, acc_class:%.2f, acc_action:%.2f " % (eval_loss_mlm, eval_loss_itm, eval_loss_order, eval_loss_class, avg_acc_mlm, avg_acc_itm, avg_acc_order, avg_acc_class, avg_acc_action))

    print(result)
    return result



def main():
    #parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Process some integers.')
    ## Required parameters
    parser.add_argument("--train_data_file", default='data/train/', type=str,  help="The input training data file (a text file).")    
    parser.add_argument("--eval_data_file", default='data/collect_traj_test/', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--output_dir", default='result/', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")    

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_trainval", action='store_true',
                        help="Whether to run eval when training.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=48, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=80.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,  #-1
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--vision_size", type=int, default=2176,help="imgaction size")
    parser.add_argument("--action_space", type=int, default=36,help="action space")
    parser.add_argument("--vl_layers", type=int, default=4,help="how many fusion layers")
    parser.add_argument("--la_layers", type=int, default=9,help="how many lang layers")
    parser.add_argument('--update', type=bool, default=True, help='update lang Bert')
    parser.add_argument('--update_add_layer', type=bool, default=True, help='update add layer')
    parser.add_argument('--include_next', type=bool, default=True, help='do action classification')
    parser.add_argument('--result_dir', type=str, default='tasks/R2R/results/', help='path to the result_dir file')
    parser.add_argument('--plot_dir', type=str, default='tasks/R2R/plots/', help='path to the plot_dir file')
    parser.add_argument('--snapshot_dir', type=str, default='tasks/R2R/snapshots/', help='path to the snapshot_dir file')
    parser.add_argument('--philly', action='store_true', help='program runs on Philly, used to redirect `write_model_path`')
    parser.add_argument("--resume_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.") 
    parser.add_argument('-r', '--run_name', type=str, help="name for wandb run", required=True)
    args = parser.parse_args()
    params = vars(args)

    if params['philly']: # use philly False
        print('Info: Use Philly, all the output folders are reset.')
        RESULT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['result_dir'])
        PLOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['plot_dir'])
        SNAPSHOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['snapshot_dir'])
        print('RESULT_DIR', RESULT_DIR)
        print('PLOT_DIR', PLOT_DIR)
        print('SNAPSHOT_DIR', SNAPSHOT_DIR)

    if params['model_type'] in ["bert", "roberta", "distilbert"] and not params['mlm']: #False
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if params['eval_data_file'] is None and params['do_eval']: 
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(params['output_dir']) and os.listdir(params['output_dir']) and params['do_train'] and not params['overwrite_output_dir']:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(params['output_dir']))

    # Setup distant debugging if needed
    if params['server_ip'] and params['server_port']:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(params['server_ip'], params['server_port']), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if params['local_rank'] == -1 or params['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not params['no_cuda'] else "cpu")
        params['n_gpu'] = torch.cuda.device_count()
        print("You are using %d GPUs to train!!" % (params['n_gpu']))
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #set_trace()
        torch.cuda.set_device(params['local_rank'])
        device = torch.device("cuda", params['local_rank'])
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.init_process_group("nccl", init_method=init_method,
                        world_size=world_size, rank=rank)
        params['n_gpu'] = 1
    params['device'] = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if params['local_rank'] in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    params['local_rank'], device, params['n_gpu'], bool(params['local_rank'] != -1), params['fp16'])

    if params['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    config_class, model_class, tokenizer_class = MODEL_CLASSES[params['model_type']]
    config = config_class.from_pretrained(params['config_name'] if params['config_name'] else params['model_name_or_path'])
    tokenizer = tokenizer_class.from_pretrained(params['tokenizer_name'] if params['tokenizer_name'] else params['model_name_or_path'], do_lower_case=params['do_lower_case'])
    if params['block_size'] <= 0: #-1
        params['block_size'] = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    params['block_size'] = min(params['block_size'], tokenizer.max_len_single_sentence)
    config.img_feature_dim = params['vision_size']
    config.img_feature_type = ""
    config.update_lang_bert = params['update']
    config.update_add_layer = params['update_add_layer']
    config.vl_layers = params['vl_layers']
    config.la_layers = params['la_layers']
    config.action_space = params['action_space']

    if params['resume_path'] is not None:
        model = DicAddActionPreTrain.from_pretrained(params['resume_path'])
        print("you have loaded model from %s" % (params['resume_path']))
    else:
        model = DicAddActionPreTrain(config)
    model.to(params['device'])

    if params['local_rank'] == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", params)

    # Training
    if params['do_train']:
        if params['local_rank'] not in [-1, 0]:
            torch.distributed.barrier()  
        jfiles = glob.glob(params['train_data_file'] + "/*.json")
        train_dataset = NavDataset(jfiles, tokenizer, feature_store, panoramic, params, feature_store_bnb)
        print("you have loaded %d  time steps" % (len(train_dataset)))

        if params['local_rank'] == 0:
            torch.distributed.barrier()
        global_step, tr_loss = train(params, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if params['do_trainval']:
        if params['local_rank'] not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        jfiles = glob.glob(params['train_data_file'] + "/*.json")  
        jfiles_bnb = 'data/bnb/traj_train.json'  
        jfiles_bnb = None    
        train_dataset = NavDataset(jfiles, jfiles_bnb, tokenizer, feature_store, panoramic, params, feature_store_bnb)
        print("you have loaded %d  time steps" % (len(train_dataset)))
        jfiles_eval = glob.glob(params['eval_data_file'] + "/*.json")
        eval_dataset = NavDataset(jfiles_eval, None, tokenizer, feature_store, panoramic, params, feature_store_bnb)
        print("you have loaded %d  time steps" % (len(eval_dataset)))

        if params['local_rank'] == 0:
            torch.distributed.barrier()
        global_step, tr_loss = trainval(params, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if params['do_train'] and (params['local_rank'] == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(params['output_dir']) and params['local_rank'] in [-1, 0]:
            os.makedirs(params['output_dir'])
        logger.info("Saving model checkpoint to %s", params['output_dir'])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(params['output_dir'])
        tokenizer.save_pretrained(params['output_dir'])
        # Good practice: save your training arguments together with the trained model
        torch.save(params, os.path.join(params['output_dir'], 'training_args.bin'))
        # Load a trained model and vocabulary that you have fine-tuned
        model = DicAddActionPreTrain.from_pretrained(params['output_dir'])
        tokenizer = tokenizer_class.from_pretrained(params['output_dir'], do_lower_case=params['do_lower_case'])
        model.to(params['device'])


    if params['do_eval']:
        jfiles = glob.glob(params['eval_data_file'] + "/*.json")
        eval_dataset = NavDataset(jfiles, tokenizer, feature_store, panoramic, params, feature_store_bnb)
        print("you have loaded %d  time steps" % (len(eval_dataset)))

        if params['local_rank'] == 0:
            torch.distributed.barrier()
        results = evaluate(params, eval_dataset, model, tokenizer)  

    results = {}
    return results
if __name__ == "__main__":
    main()
