# Computes metrics

# v1  -> Bug corrected in call to compute metrics
# v2  -> Plots learning curves
# v3  -> Reads result in jston format. Input path updated
# v4  -> Converts to multilabel classification task and computes metrics

#%% Imports
import os
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#%% Path definitions
base_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/15_LegalPredictor_v1/00_data/03_runs/00_multilabel/01_selected_ECHR_arts/01_BERT_TRANSF_v2_FIX_50par_6ep'

#%% Global initialization
random.seed(1234)
threshold = 0.5
tgt_labels = ['2', '3', '4', '5', '6', '7', '8', '9',
              '10', '12', '13', '14', '15', '17', '18']

#%% Read data json
input_path = os.path.join(base_path, 'full_results_model_dev.json')
#input_path = os.path.join(base_path, 'full_results_dev.json')
with open(input_path) as fr:
    results = json.load(fr)

#%% Extract results
Y_pred_score = results['Y_test_prediction_scores']
Y_ground_truth = results['Y_test_ground_truth']
Y_pred_binary = [[1 if x >= threshold else 0 for x in sublist] for sublist in Y_pred_score]
Y_ground_truth = [[int(x) for x in sublist] for sublist in Y_ground_truth]

#%% Print results    
print(classification_report(Y_ground_truth, Y_pred_binary,
      target_names = tgt_labels))

#%% Plot learning curves
plt.plot(results['training_loss'], label = 'train')
plt.plot(results['validation_loss'], label = 'validation')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()

