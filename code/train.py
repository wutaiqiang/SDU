import argparse
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import os
import pickle
import torch
import torch.distributed as dist
import sys
import io
import random
import time
import numpy as np
import sklearn.metrics as tool

from model import Bert_promopt
from optimizer import get_bert_optimizer
from pytorch_transformers.optimization import WarmupLinearSchedule



# 获得id:[XX,XX,XX]的dict
def read_id_candidate(params,mode,diction):
    fpath = os.path.join(params["input_path"],mode+'.json')
    with open(fpath,encoding='utf-8') as file:
        data = json.load(file)
    D = {}
    for i,d in enumerate(data):
        #s = d["sentence"].split(d["ID"])
        candidate = diction[d["acronym"]]
        D[d["ID"]] = candidate
    return D
    
def generate(reranker, eval_dataloader, device, D=None, out_path=None):
    reranker.model.eval() #eval模式
    iter_ = tqdm(eval_dataloader, desc="Evaluation")

    all_ids = []
    all_pred = []
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        if len(batch) == 4:
            context, ids, lens, label = batch
        else:
            context, ids, lens = batch

        with torch.no_grad():
            logits = reranker(context, label=None, lens=None)

        logits = logits.detach().cpu().numpy()
        #outputs = np.argmax(out, axis=1)
        #all_pred.extend(np.argmax(logits, axis=1))
        lens = lens.detach().cpu().numpy()
        #outputs = np.argmax(out, axis=1)
        for i,logit in enumerate(logits):
            all_pred.append(np.argmax(logit[:lens[i]]))

        all_ids.extend(ids.detach().cpu().numpy())
    if out_path:
        # 检索对应的字符
        d = [{'ID': str(all_ids[i]),'label': D[str(all_ids[i])][all_pred[i]]} for i in range(len(all_ids))]
        with open(out_path, 'w',encoding='utf-8') as file:
            json.dump(d,file,ensure_ascii=False) #! For fr/sp

def evaluate(reranker, eval_dataloader, device):
    reranker.model.eval() #eval模式
    iter_ = tqdm(eval_dataloader, desc="Evaluation")

    all_label = []
    all_pred = []
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context, _, lens, label = batch
        with torch.no_grad():
            logits = reranker(context, lens=None, label=None)

        logits = logits.detach().cpu().numpy()
        all_label.extend(label.detach().cpu().numpy())
        lens = lens.detach().cpu().numpy()
        #outputs = np.argmax(out, axis=1)
        for i,logit in enumerate(logits):
            all_pred.append(np.argmax(logit[:lens[i]]))
        #all_pred.extend(np.argmax(logits, axis=1))
    print('precision: ',tool.precision_score(all_label,all_pred,average='macro'))
    print('f1 score: ',tool.f1_score(all_label,all_pred,average='macro'))
    print('recall: ',tool.recall_score(all_label,all_pred,average='macro'))
    


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )

def get_scheduler(params, optimizer, len_train_data):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    return scheduler

def read_dataset_to_tensor(params,tokenizer,with_label,mode,diction):
    fpath = os.path.join(params["input_path"],mode+'.json')

    try:
        with open(fpath,encoding='utf-8') as file:
            data = json.load(file)
    except:
        print("No file:", fpath)
    # 处理数据
    if with_label:
        label_idx_all = []
    token_all = []
    IDs = []

    # 存储信息
    lens = [len(diction[d["acronym"]]) for d in data]
    max_len = max(lens)+2 #至少补两个hard sample
    all_candidate = []
    for k,v in diction.items():
        all_candidate.extend(v)

    print("processing ", mode)

    #! For debug
    if params["debug"]:
        data = data[:100]
        lens = lens[:100]

    for i,d in enumerate(tqdm(data)):
        tokens = []
        #s = d["sentence"].split(d["ID"])
        candidate = diction[d["acronym"]]
        if with_label:
            label_idx_all.append(candidate.index(d["label"]))
        sen_id = tokenizer.encode(d["sentence"])
        if len(candidate) < max_len: #随机负采样
            for _ in range(max_len-len(candidate)):
                index = np.random.randint(low=0,high=len(all_candidate)-1)
                candidate.append(all_candidate[index])
        assert len(candidate) == max_len, "Not padded!"
        for c in candidate:
            if params["language"] == "en":
                ids = sen_id + [102] + tokenizer.encode(c) + [102] +  \
                    tokenizer.encode("the meaning of " + d["acronym"] + " is or equals " + c) # ** [sep] ***
            elif params["language"] == "sp":
                ids = sen_id + [102] + tokenizer.encode(c) + [102] +  \
                    tokenizer.encode("el significado de " + d["acronym"] + " es o igual a " + c) # ** [sep] ***
            elif params["language"] == "fr":
                ids = sen_id + [102] + tokenizer.encode(c) + [102] +  \
                    tokenizer.encode("la signification de " + d["acronym"] + " est ou est égale à " + c) # ** [sep] ***

            tokens.append(ids+[0]*(params["max_seq_len"]-len(ids)) if len(ids)<params["max_seq_len"] else ids[-params["max_seq_len"]:])
        token_all.append(tokens)
        IDs.append(int(d["ID"]))
    print("processing done: ", mode)
    # To tensor
    token_all = torch.tensor(token_all, dtype=torch.long,)
    if with_label:
        label_idx_all = torch.tensor(label_idx_all, dtype=torch.long,)
    IDs = torch.tensor(IDs, dtype=torch.long,)
    lens = torch.tensor(lens, dtype=torch.long,)

    if with_label:
        return TensorDataset(token_all,IDs,lens,label_idx_all)
    else:
        return TensorDataset(token_all,IDs,lens)


def main(params):
    # 输出目录
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # initial model
    reranker = Bert_promopt(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device

    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    #! Read data
    with open(os.path.join(params["input_path"],'diction.json'),encoding='utf-8') as file:
        diction = json.load(file)

    train_tensor_data = read_dataset_to_tensor(params,tokenizer,with_label=True,mode='train',diction=diction)
    
    # 如果shuffle就 random sampler
    if params["not_shuffle"]:
        train_sampler = SequentialSampler(train_tensor_data)
    else:
        train_sampler = RandomSampler(train_tensor_data) # if shuffle
    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size,
    )

    valid_tensor_data = read_dataset_to_tensor(params,tokenizer,with_label=True,mode='dev',diction=diction)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size,
    )

    test_tensor_data = read_dataset_to_tensor(params,tokenizer,with_label=False,mode='test',diction=diction)
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=eval_batch_size,
    )

    # evaluate
    evaluate(reranker, valid_dataloader, device)

    #D = read_id_candidate(params,"dev",diction)
    #generate(reranker, valid_dataloader, device, D, "/apdcephfs/private_takiwu/SDU/1.json")

    # 初始化
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data))

    num_train_epochs = params["num_train_epochs"]
    #! Add, load from check point
    if params["path_to_model"] == None:
        start_epoch = 0
    else:
        t = params["path_to_model"].split('/')[-2] #epoch_**
        assert t.split('_')[0]=='epoch', print("Wrong checkpoint dir")
        start_epoch = int(t.split('_')[-1]) + 1
        print("Continue training at epoch: "+ str(start_epoch))
    
    # train
    model.train()
    for epoch_idx in trange(start_epoch, int(num_train_epochs), desc="Epoch"):

        tr_loss = 0

        iter_ = tqdm(train_dataloader, desc="Batch")
        
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context, _, lens, label = batch
            loss = reranker(context, label=label, lens=lens)

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                print(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                print("Evaluation on the development dataset")
                evaluate(reranker, valid_dataloader, device)             
                model.train()
        
        print("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        #reranker.save_model(epoch_output_folder_path)
        if not os.path.exists(epoch_output_folder_path):
            os.makedirs(epoch_output_folder_path)

        file_out_path1 = os.path.join(epoch_output_folder_path,'valid_out.json')
        file_out_path2 = os.path.join(epoch_output_folder_path,'test_out.json')
        
        #! 传入的是 DEV的数据映射
        D1 = read_id_candidate(params,"dev",diction)
        generate(reranker, valid_dataloader, device, D1, file_out_path1)

        D2 = read_id_candidate(params,"test",diction)
        generate(reranker, test_dataloader, device, D2, file_out_path2)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',  type=str,default='', help='Path of input file')
    parser.add_argument('--output_path', type=str,default='', help='Path of output file')

    parser.add_argument('--train_batch_size', type=int, default=2, help='train_batch_size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='eval_batch_size')
    parser.add_argument('--seed', type=int, default=10086, help='seed')
    parser.add_argument('--not_shuffle', action='store_true', help='Not shuffle train_set')
    parser.add_argument('--no_cuda', action='store_true', help='Not use cuda')
    parser.add_argument('--data_parallel', action='store_true', help='data parallel')
    parser.add_argument('--debug', action='store_true', help='debug')

    parser.add_argument('--print_interval', type=int, default=10, help='print_interval')
    parser.add_argument('--eval_interval', type=int, default=40, help='eval_interval')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max_seq_len')
    parser.add_argument("--gap",default=0.1,type=float,help="the gap for max margin loss",)
    parser.add_argument("--mu",default=0.5,type=float,help="loss weight",)
    

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_grad_norm', type=int, default=1, help='max_grad_norm')
    
    parser.add_argument('--language', type=str, default="en", help='language')
    #parser.add_argument('--bert_model', type=str, default="bert-base-multilingual-cased", help='bert version')
    parser.add_argument('--bert_model', type=str, default="bert-large-cased", help='bert version')
    parser.add_argument('--lowercase', action='store_true', help=" default false")
    # parser.add_argument('--cash',  type=str,default="", help='cash for huggingface')
    parser.add_argument('--path_to_model', type=str, default=None, help='path store the checkpoint of model')
    
    parser.add_argument("--type_optimization",type=str,default="all_encoder_layers",help="Which type of layers to optimize in BERT",)
    parser.add_argument("--learning_rate",default=3e-5,type=float,help="The initial learning rate for Adam.",)
    parser.add_argument("--num_train_epochs",default=5,type=int,help="Number of training epochs.",) 
    parser.add_argument("--warmup_proportion",default=0.1,type=float,help="Proportion of training to perform linear learning rate warmup for",)

    args = parser.parse_args()
    params = args.__dict__

    #params["debug"]=True
    print(params)
    
    main(params)




