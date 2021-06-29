# v0
# v1 -> Saves full checkpoints

#%% Imports
import os
import random
import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_v0 import ECHR2_dataset, ECHR2_model
import utils.utils as utils

#%% Function definitions
def run_epoch_f(args, mode, model, criterion, optimizer,
                logger, data_loader, device, epoch):
   
    if mode == 'Train':
        model.train()
        mode_desc = 'Training_epoch'
    if mode == 'Val':
        model.eval()
        mode_desc = 'Validating_epoch'

    sum_correct = 0
    total_entries = 0
    sum_loss = 0
    
    for step_idx, (X_facts_ids, X_facts_token_types, X_facts_attn_masks,
                   X_echr_ids, X_echr_token_types, X_echr_attn_masks, Y_labels) in \
                   tqdm(enumerate(data_loader), total = len(data_loader), desc = mode_desc):
        
        # Move data to cuda
        if next(model.parameters()).is_cuda:
            X_facts_ids = X_facts_ids.to(device)
            #X_facts_token_types = X_facts_token_types.to(device)
            X_facts_attn_masks = X_facts_attn_masks.to(device)
            X_echr_ids = X_echr_ids.to(device)
            #X_echr_token_types = X_echr_token_types.to(device)
            X_echr_attn_masks = X_echr_attn_masks.to(device)
            Y_labels = Y_labels.to(device)
        
        # Zero gradients
        if mode == 'Train':
            optimizer.zero_grad()
        
        #Forward + backward + optimize
        pred = model(X_facts_ids, X_facts_token_types, X_facts_attn_masks,
                     X_echr_ids, X_echr_token_types, X_echr_attn_masks)
        pred = pred.view(-1)
        loss = criterion(pred, Y_labels)
        
        if mode == 'Train':
            # Backpropagate
            loss.backward()
            # Update model
            optimizer.step()           
        
        # Book-keeping
        current_batch_size = X_facts_ids.size()[0]
        total_entries += current_batch_size
        sum_loss += (loss.item() * current_batch_size)
        pred = torch.round(pred)
        sum_correct += (pred == Y_labels).sum().item()
      
        # Log train step
        if mode == 'Train':
            logger.info(f'Epoch {epoch + 1} of {args.n_epochs}' +
                        f' Step {step_idx + 1:,} of {len(data_loader):,}')
        
    # Compute metrics
    avg_loss = sum_loss / total_entries
    avg_acc = sum_correct / total_entries
    
    # Log results
    if mode == 'Train':
        print(f'\nTrain loss: {avg_loss:.4f} and accuracy: {avg_acc:.4f}')
        logger.info(f'Train loss: {avg_loss:.4f} and accuracy: {avg_acc:.4f}')
    if mode == 'Val':
        print(f'\nValidation loss: {avg_loss:.4f} and accuracy: {avg_acc:.4f}')
        logger.info(f'Validation loss: {avg_loss:.4f} and accuracy: {avg_acc:.4f}')
    
    return avg_loss, avg_acc

#%% Main
def main():
    # Arg parsing
    args = utils.parse_args_f()
    
    # Path initialization train-dev
    path_model_train = os.path.join(args.input_dir, 'model_train.pkl')
    path_model_dev = os.path.join(args.input_dir, 'model_dev.pkl')
    
    # Create ouput dir
    utils.make_dir_f(args)
    
    # Instantiate logger
    logger = utils.get_logger_f(args)
      
    # Global and seed initialization
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    random.seed = args.seed
    _ = torch.manual_seed(args.seed)

    # Load train dev test sets
    print('Loading data')
    model_train = pd.read_pickle(path_model_train)
    model_dev = pd.read_pickle(path_model_dev)
    print('Done')
    
    if eval(args.train_toy_data) == True:
        model_train = model_train[0:args.len_train_toy_data]
        model_dev = model_dev[0:args.len_train_toy_data]
   
    # Instantiate dataclasses
    train_dataset = ECHR2_dataset(model_train)
    dev_dataset = ECHR2_dataset(model_dev)

    # Instantiate dataloaders
    train_dl = DataLoader(train_dataset, batch_size = args.batch_size,
                          shuffle = eval(args.shuffle_train),
                          drop_last = eval(args.drop_last_train))
    dev_dl = DataLoader(dev_dataset,
                        batch_size = int(args.batch_size * args.dev_train_ratio),
                        shuffle = False)

    # Instantiate model
    model = ECHR2_model(args)

    # Set device and move model to device
    model, device = utils.model_2_device_f(args, model)

    # Instantiate optimizer & criterion
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.lr,
                                 weight_decay = args.wd)
    criterion = nn.BCELoss()

    # Save model parameters    
    utils.save_args_f(args)

    # Training procedure
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    start_time = datetime.datetime.now()
    
    for epoch in tqdm(range(args.n_epochs), desc = 'Training dataset'):
    
        mode = 'Train'    
        train_loss, train_acc = run_epoch_f(args, mode, model, criterion,
                                            optimizer, logger, train_dl,                                              
                                            device, epoch)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        mode = 'Val'
        val_loss, val_acc = run_epoch_f(args, mode, model, criterion,
                                        optimizer, logger, dev_dl,
                                        device, epoch)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc) 

        # Save checkpoint
        if eval(args.save_model_steps) == True and epoch >= args.save_step_cliff:
            utils.save_checkpoint_f(args, epoch, model, optimizer, train_loss)
    
    # Save model
    if eval(args.save_final_model) == True:
        utils.save_checkpoint_f(args, epoch, model, optimizer, train_loss)
    
    # Save results
    utils.save_results_f(args, train_loss_history, train_acc_history,
                         val_loss_history, val_acc_history, start_time)

if __name__ == "__main__":
    main()
