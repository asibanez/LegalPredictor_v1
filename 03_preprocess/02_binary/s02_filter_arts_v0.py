# Filters examples with unwanted articles

#%% Imports
import os
import pandas as pd

#%% Function definitions
def filter_f(dataset, num_total_arts, selected_positions):
    filtering = [True if x in selected_positions else False for x in range(num_total_arts)]
    filtering = filtering * (int(len(dataset) / num_total_arts))
    dataset = dataset[filtering]
    return dataset

#%% Global initialization
id_2_label = {0: '2',
              1: '3',
              2: '4',
              3: '5',
              4: '6',
              5: '7',
              6: '8',
              7: '9',
              8: '10',
              9: '11',
              10:'12',
              11:'13',
              12:'14',
              13:'15',
              14:'17',
              15:'18',
              16:'34',
              17:'38',
              18:'39',
              19:'46',              
              20:'P1-1',
              21:'P1-2',
              22:'P1-3',
              23:'P4-2',
              24:'P4-4',
              25:'P6-3',
              26:'P7-1',
              27:'P7-2',
              28:'P7-3',
              29:'P7-4',
              30:'P7-5',
              31:'P12-1'} # 'P3-1' removed

selected_arts = ['2', '3', '4', '5', '6', '7', '8', '9',
                     '10', '12', '13', '14', '15', '17', '18']

#%% Path definition
#input_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/15_LegalPredictor_v1/00_data/02_preprocessed/01_binary/01_selected_echr_arts/00_50pars_256_tok'
input_folder = '/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/02_preprocessed/01_binary/01_selected_ECHR_arts'
output_folder = os.path.join(input_folder,'00_filtered')

input_train_set_path = os.path.join(input_folder, 'model_train.pkl')
input_dev_set_path = os.path.join(input_folder, 'model_dev.pkl')
input_test_set_path = os.path.join(input_folder, 'model_test.pkl')

output_train_set_path = os.path.join(output_folder, 'model_train.pkl')
output_dev_set_path = os.path.join(output_folder, 'model_dev.pkl')
output_test_set_path = os.path.join(output_folder, 'model_test.pkl')

#%% Data load
train_set = pd.read_pickle(input_train_set_path)
dev_set = pd.read_pickle(input_dev_set_path)
test_set = pd.read_pickle(input_test_set_path)

#%% Filter datasets
num_total_arts = len(id_2_label)
label_2_id = {id_2_label[x]:x for x in id_2_label.keys()}
selected_positions = [label_2_id[x] for x in selected_arts]

train_set = filter_f(train_set, num_total_arts, selected_positions)
dev_set = filter_f(dev_set, num_total_arts, selected_positions)
test_set = filter_f(test_set, num_total_arts, selected_positions)

#%% Re-index dataframes
train_set = train_set.reset_index(drop = True)
dev_set = dev_set.reset_index(drop = True)
test_set = test_set.reset_index(drop = True)

#%% Save datasets
print('Saving datasets')
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    print("Created folder : ", output_folder)
pd.to_pickle(train_set, output_train_set_path)
pd.to_pickle(dev_set, output_dev_set_path)
pd.to_pickle(test_set, output_test_set_path)
print('Done')
