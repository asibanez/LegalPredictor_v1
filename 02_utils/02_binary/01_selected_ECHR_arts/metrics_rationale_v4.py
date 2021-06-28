# v1  -> Bug corrected in call to compute metrics
# v2  -> Plots learning curves
# v3  -> Reads result in jston format. Input path updated
# v4  -> Converts to multilabel classification task and computes metrics

#%% Imports
import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#%% Path definitions
base_path_run = 'C:/Users/siban/Dropbox/CSAIL/Projects/15_LegalPredictor_v1/00_data/03_runs/01_binary/01_selected_ECHR_arts/00_BERT_TRANSF_v2_FIX_50par_10ep'
path_raw = 'C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_50pars_256_tok/01_full'

#%% Global initialization
random.seed(1234)
threshold = 0.5
tgt_labels = ['2',
              '3',
              '4',
              '5',
              '6',
              '7',
              '8',
              '9',
              '10',
              '11',
              '12',
              '13',
              '14',
              '15',
              '17',
              '18',
              '34',
              '38',
              '39',
              '46',              
              'P1-1',
              'P1-2',
              'P1-3',
              'P3-1',
              'P4-2',
              'P4-4',
              'P6-3',
              'P7-1',
              'P7-2',
              'P7-3',
              'P7-4',
              'P7-5',
              'P12-1']

selected_arts = ['2', '3', '4', '5', '6', '7', '8', '9',
                     '10', '12', '13', '14', '15', '17', '18']

#%% Define data path
results_path = os.path.join(base_path_run, 'full_results_model_test.json')
#results_path = os.path.join(base_path_run, 'train_results.json')

#%% Load datasets
with open(results_path) as fr:
    results = json.load(fr)

ground_truth = load_dataset('ecthr_cases', split = 'validation')
ground_truth = load_dataset('ecthr_cases', split = 'test')

#%% Extract info
Y_pred_score = results['Y_test_prediction_scores']
Y_pred_binary = results['Y_test_prediction_binary']
Y_g_truth_binary = results['Y_test_ground_truth']

#%% Plot learning curves
plt.plot(results['training_loss'], label = 'train')
plt.plot(results['validation_loss'], label = 'validation')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()

#%% De-binarize results
num_selected_arts = len(selected_arts)
num_examples_multilabel = int(len(Y_pred_score) / num_selected_arts)
Y_g_truth_multilabel = []
Y_pred_multilabel = []

# Fix Y_pred binary
Y_pred_binary = [x[0] for x in Y_pred_binary]


for idx in range(num_examples_multilabel):
    sel_b = idx * num_selected_arts
    sel_e = (idx + 1) * num_selected_arts
    Y_g_truth_multilabel.append(Y_g_truth_binary[sel_b:sel_e])
    Y_pred_multilabel.append(Y_pred_binary[sel_b:sel_e])

#%% Compute classification results
print(classification_report(Y_g_truth_multilabel, Y_pred_multilabel,
      target_names = selected_arts))
