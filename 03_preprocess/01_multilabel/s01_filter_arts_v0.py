# Filters uwanted articles from labels

#%% Imports
import os
import pandas as pd

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
              23:'P3-1',
              24:'P4-2',
              25:'P4-4',
              26:'P6-3',
              27:'P7-1',
              28:'P7-2',
              29:'P7-3',
              30:'P7-4',
              31:'P7-5',
              32:'P12-1'}

selected_articles = ['2', '3', '4', '5', '6', '7', '8', '9',
                     '10', '12', '13', '14', '15', '17', '18']

#%% Path definition
#work_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/15_LegalPredictor_v1/00_data/02_preprocessed/00_multilabel/01_selected_echr_arts'
work_folder = '/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/02_preprocessed/00_multilabel/01_selected_ECHR_arts'
input_train_set_path = os.path.join(work_folder, 'model_train.pkl')
input_dev_set_path = os.path.join(work_folder, 'model_dev.pkl')
input_test_set_path = os.path.join(work_folder, 'model_test.pkl')

output_train_set_path = os.path.join(work_folder, 'model_train_filtered.pkl')
output_dev_set_path = os.path.join(work_folder, 'model_dev_filtered.pkl')
output_test_set_path = os.path.join(work_folder, 'model_test_filtered.pkl')

#%% Data load
train_set = pd.read_pickle(input_train_set_path)
dev_set = pd.read_pickle(input_dev_set_path)
test_set = pd.read_pickle(input_test_set_path)

#%% Preprocess labels
print('Preprocessing labels')
label_2_id = {id_2_label[x]:x for x in id_2_label.keys()}
selected_positions = [label_2_id[x] for x in selected_articles]
print('Done')

labels_train = train_set.labels
labels_dev = dev_set.labels
labels_test = test_set.labels

labels_train_new = [[label[x] for x in selected_positions] for label in labels_train]
labels_dev_new = [[label[x] for x in selected_positions] for label in labels_dev]
labels_test_new = [[label[x] for x in selected_positions] for label in labels_test]

train_set.labels = labels_train_new
dev_set.labels = labels_dev_new
test_set.labels = labels_test_new

#%% Save datasets
print('Saving datasets')
pd.to_pickle(train_set, output_train_set_path)
pd.to_pickle(dev_set, output_dev_set_path)
pd.to_pickle(test_set, output_test_set_path)
print('Done')
