# v2 -> Tries to use stacked tensors
# v3 -> Uses flat tensors

# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

#%% DataClass definition
class ECHR2_dataset(Dataset):
    def __init__(self, data_df):
        self.facts_ids = data_df['facts_ids']
        self.facts_token_types = data_df['facts_token_type']
        self.facts_attn_masks = data_df['facts_att_mask']
        self.labels = np.asarray(data_df.labels.to_list())
        
    def __len__(self):
        return len(self.facts_ids)
        
    def __getitem__(self, idx):
        X_facts_ids = self.facts_ids[idx]
        X_facts_token_types = self.facts_token_types[idx]
        X_facts_attn_maks = self.facts_attn_masks[idx]
        Y_labels = self.labels[idx]
        
        return X_facts_ids, X_facts_token_types, X_facts_attn_maks, Y_labels

#%% Model definition
class ECHR2_model(nn.Module):
            
    def __init__(self, args):
        super(ECHR2_model, self).__init__()

        #self.max_n_pars = 200
        self.max_n_pars = 20
        self.h_dim = 512
        self.n_heads = 8
        self.n_labels = 33
        self.seq_len = 512
        self.dropout = 0.4 #args.dropout
                     
        # Bert model
        self.model_name = 'nlpaueb/legal-bert-small-uncased'
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        
        # Transformer layer
        self.transf_enc = nn.TransformerEncoderLayer(d_model = self.h_dim,
                                                     nhead = self.n_heads)
    
        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim,
                                out_features = self.n_labels)

        # Pooling
        self.max_pool = nn.MaxPool1d(kernel_size = self.max_n_pars)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
            
    def forward(self, X_facts_ids, X_facts_token_types, X_facts_attn_masks):
        batch_size = X_facts_ids.size()[0]
        
        # Encode paragraphs - BERT & generate transfomers masks
        bert_out = {}
        transf_mask = torch.zeros((batch_size,
                                   self.max_n_pars), dtype=torch.bool)  # batch_size x max_n_pars
        
        for idx in tqdm(range(0, self.max_n_pars)):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            
            # Slice sequence
            facts_ids = X_facts_ids[:, span_b:span_e]                   # batch_size x seq_len
            facts_token_types = X_facts_token_types[:, span_b:span_e]   # batch_size x seq_len
            facts_attn_masks = X_facts_attn_masks[:, span_b:span_e]     # batch_size x seq_len
            
            # Generate masks for transformer
            mask_aux = torch.sum(facts_ids, dim = 1).view(-1,1)         # batch_size x 1
            mask_aux = (mask_aux == 0)                                  # batch_size x 1
            transf_mask[:, idx] = mask_aux.view(1,-1)                   # 1 x batch_size
            
            # Generate input dict to bert model
            bert_input = {'input_ids': facts_ids,
                          'token_type_ids': facts_token_types,
                          'attention_mask': facts_attn_masks}
                  
            # Compute 
            output = self.bert_model(**bert_input, output_hidden_states = True)
            bert_out[idx] = output['pooler_output'].unsqueeze(1)        # batch_size x 1 x h_dim
        
        x = torch.cat(list(bert_out.values()), dim=1)                   # batch_size x max_n_pars x h_dim
        
        # Encode document - Transformer
        x = x.transpose(0, 1)                                           # max_n_pars x batch_size x h_dim
        x = self.transf_enc(x,src_key_padding_mask=transf_mask)         # max_n_pars x batch_size x h_dim
        x = x.transpose(0, 1)                                           # batch_size x max_n_pars x h_dim
        
        # Max pooling over paragraphs
        x = x.transpose(1,2)                                            # batch_size x h_dim x max_n_pars
        x = self.max_pool(x)                                            # batch_size x h_dim x 1
        
        # Multi-label classifier
        x = x.squeeze(2)                                                # batch_size x h_dim
        x = self.fc_out(x)                                              # batch_size x n_lab
        x = self.sigmoid(x)                                             # batch_size x n_lab

        return x

#%% Load data
train_set = pd.read_pickle('C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/03_toy_3/model_train.pkl')

#%% Instantiate dataclass & dataloader
train_dataset = ECHR2_dataset(train_set)
train_dl = DataLoader(train_dataset, batch_size = 2,
                      shuffle = False, drop_last = True)

#%% Instantiate model
ECHR2_model = ECHR2_model(None)

#%% Compute predictions
for X_facts_ids, X_facts_token_types, X_facts_attn_masks, Y_labels in train_dl:
    pred = ECHR2_model(X_facts_ids, X_facts_token_types, X_facts_attn_masks)