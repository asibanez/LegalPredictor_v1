# Converts echr arts and facts ids to text
# Verifies consistency of preprocessed dataset
# Visualization of full examples

#%% Imports
import os
import torch
import pandas as pd
from transformers import AutoTokenizer

#%% Path definition
input_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/15_LegalPredictor_v1/00_data/02_preprocessed/01_binary/01_selected_echr_arts/00_50pars_256_tok/filtered'
#input_folder = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/00_full'

train_set_path = os.path.join(input_folder, 'model_train.pkl')
val_set_path = os.path.join(input_folder, 'model_dev.pkl')
test_set_path = os.path.join(input_folder, 'model_test.pkl')

#%% Global initialization
pos = 0
max_num_pars_facts = 50
max_num_pars_echr = 6
seq_len = 256
num_selected_arts = 15

#%% Data load
print('Loading dataset')
dataset = pd.read_pickle(train_set_path)
dataset = dataset.reset_index(drop = True)
print('Done')
len_dataset = len(dataset)

#%% Tokenizer instantiation
model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

#%% Verify ECHR articles consistent across dataset
for pos in range(num_selected_arts):
    # Generate slicing vector
    filtering = [False] * num_selected_arts
    filtering[pos] = True
    filtering = filtering * int((len_dataset / num_selected_arts))
    # Check if all items equal
    echr_token_ids = dataset['echr_ids']
    echr_token_ids = echr_token_ids[filtering]
    echr_token_ids = [x.tolist() for x in echr_token_ids]
    all_entries_equal = all([x == echr_token_ids[0] for x in echr_token_ids])
    
    # Convert ECHR arts ids to text
    echr_par_text_list = []
    echr_token_ids = dataset.iloc[pos]['echr_ids']
    for par_idx in range(max_num_pars_echr):
        par_b = seq_len * par_idx
        par_e = seq_len * (par_idx + 1)
        par_token_ids = echr_token_ids[par_b:par_e]
        # Remove [PAD] token IDs
        par_token_ids = [token_id for token_id in par_token_ids if token_id != 0]
        par_token_ids = torch.tensor(par_token_ids)
        par_text = bert_tokenizer.decode(par_token_ids)
        echr_par_text_list.append(par_text)
    
    # Print results
    print('#' * 60)
    print(f'Pos = {pos}')
    print(f'\nAll entries equal = {all_entries_equal}')
    print('\nECHR art text:')
    for x in echr_par_text_list: print(f'{x}\n')

#%% Visualize full example
############ Set desired position
# pos = example * 15 + article
pos = 8999 * 15 + 4
############
echr_id = dataset.iloc[pos]['echr_identifiers']
echr_token_ids = dataset.iloc[pos]['echr_ids']
facts_id = dataset.iloc[pos]['facts_identifiers']
facts_token_ids = dataset.iloc[pos]['facts_ids']
label = dataset.iloc[pos]['label']

# Convert ECHR arts
echr_par_text_list = []
for par_idx in range(max_num_pars_echr):
    par_b = seq_len*par_idx
    par_e = seq_len*(par_idx + 1)
    par_token_ids = echr_token_ids[par_b:par_e]
    # Remove [PAD] token IDs
    par_token_ids = [token_id for token_id in par_token_ids if token_id != 0]
    par_token_ids = torch.tensor(par_token_ids)
    par_text = bert_tokenizer.decode(par_token_ids)
    echr_par_text_list.append(par_text)

# Convert facts
facts_par_text_list = []
for par_idx in range(max_num_pars_facts):
    par_b = seq_len*par_idx
    par_e = seq_len*(par_idx + 1)
    par_token_ids = facts_token_ids[par_b:par_e]
    # Remove [PAD] token IDs
    par_token_ids = [token_id for token_id in par_token_ids if token_id != 0]
    par_token_ids = torch.tensor(par_token_ids)
    par_text = bert_tokenizer.decode(par_token_ids)
    facts_par_text_list.append(par_text)

# Print results
print(f'Example ID = {facts_id}')
print(f'ECHR art ID = {echr_id}')
print(f'label = {label}')
print('\nECHR art text:' + ' ' * 10 + '#' * 20 + '\n')
for x in echr_par_text_list: print(f'{x}\n')
print('\nfacts text:' + ' ' * 10 + '#' * 20 + '\n')
for x in facts_par_text_list: print(f'{x}\n')

