# Converts ECHR_v1 datasets in csv files

#%% Imports
import os
import pandas as pd
from datasets import load_dataset

#%% Path definition
base_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/15_LegalPredictor_v1/00_data/01_original'
output_path_train = os.path.join(base_path, 'train.csv')
output_path_dev = os.path.join(base_path, 'dev.csv')
output_path_test = os.path.join(base_path, 'test.csv')

#%% Dataload
train_set = load_dataset('ecthr_cases', split = 'train')
dev_set = load_dataset('ecthr_cases', split = 'validation')
test_set = load_dataset('ecthr_cases', split = 'test')

#%% Convert to dataframes
train_set = pd.DataFrame(train_set)
dev_set = pd.DataFrame(dev_set)
test_set = pd.DataFrame(test_set)

#%% Filter columns
train_set = train_set[['labels', 'facts']]
dev_set = dev_set[['labels', 'facts']]
test_set = test_set[['labels', 'facts']]

#%% Save as csv
train_set.to_csv(output_path_train)
dev_set.to_csv(output_path_dev)
test_set.to_csv(output_path_test)

