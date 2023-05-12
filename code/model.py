import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizer

from transformers import BertTokenizer, BertModel

from pytorch_transformers.modeling_utils import WEIGHTS_NAME,CONFIG_NAME
from transformers.models import bert

class Bert_module(torch.nn.Module):
    def __init__(self, params):
        super(Bert_module, self).__init__()
        self.model_type = params["bert_model"]
        self.bert_model = BertModel.from_pretrained(params["bert_model"])
        #self.bert_model = CamembertModel.from_pretrained(params["bert_model"])
        dim = self.bert_model.config.hidden_size
        self.additional_linear = nn.Linear(dim,1)
        self.config = self.bert_model.config

    def forward(self, input):
        #print(self.model_type)
        if self.model_type == "bert-base-multilingual-cased":
            A = self.bert_model(input)
            bert_pooler = A.pooler_output
        else: 
            bert_pooler = self.bert_model(input)[1]
        
        #print(bert_pooler)
        out = self.additional_linear(bert_pooler)
        return out.squeeze()


class Bert_promopt(torch.nn.Module):
    def __init__(self, params):
        super(Bert_promopt, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], 
            do_lower_case=params["lowercase"]
        )
        # self.tokenizer = CamembertTokenizer.from_pretrained(
        #     params["bert_model"], 
        #     do_lower_case=params["lowercase"]
        # )
        self.model = Bert_module(params)
        
        model_path = params.get("path_to_model", None)
        #print("path_to_model", model_path)
        if model_path is not None: # 加载训练好的权重
            print("loading model from {}".format(model_path))
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict, strict=False)
        
        # To Cuda
        self.model = self.model.to(self.device)
        # For DataParallel
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel: # DP 并行
            self.model = torch.nn.DataParallel(self.model)

        self.loss_fuc = nn.CrossEntropyLoss()
        self.gap = params["gap"]
        self.mu = params["mu"]

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)

    def forward(self, seq, label=None, lens=None):
        # seq: [batch, sample 数, sequence_length], label
        bsz = seq.size(0)
        seq = seq.view(-1,seq.size(-1))
        score = self.model(seq)
        score = score.view(bsz,-1) # [batch,sample num]
        if label == None:
            return score
        else:
            loss = self.loss_fuc(score,label)
        
            score = score.softmax(dim=-1)
            for index,s in enumerate(score):
                t = lens[index]
                loss += self.mu*max( max(s[t:]) + self.gap - min(s[:t]), 0)
            return loss
