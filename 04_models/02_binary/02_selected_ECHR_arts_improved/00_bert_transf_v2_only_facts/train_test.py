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
    elif mode == 'Validation':
        model.eval()
        mode_desc = 'Validating_epoch'
    elif mode == 'Test':
        model.eval()
        mode_desc = 'Testing_epoch'

    sum_correct = 0
    total_entries = 0
    sum_loss = 0
    Y_pred_score = []
    Y_pred_binary = []
    Y_gr_truth = []
    
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
        
        # Train step
        if mode == 'Train':
            # Zero gradients
            optimizer.zero_grad()
            #Forward + backward + optimize
            pred_score = model(X_facts_ids, X_facts_token_types,
                               X_facts_attn_masks, X_echr_ids,
                               X_echr_token_types, X_echr_attn_masks)
            # Compute loss
            pred_score = pred_score.view(-1)
            loss = criterion(pred_score, Y_labels)
            # Backpropagate
            loss.backward()
            # Update model
            optimizer.step()
        
        # Eval / Test step        
        else:
            with torch.no_grad(): 
                pred_score = model(X_facts_ids, X_facts_token_types,
                                   X_facts_attn_masks, X_echr_ids,
                                   X_echr_token_types, X_echr_attn_masks)
                # Compute loss
                pred_score = pred_score.view(-1)
                loss = criterion(pred_score, Y_labels)
        
        # Book-keeping
        current_batch_size = X_facts_ids.size()[0]
        total_entries += current_batch_size
        sum_loss += (loss.item() * current_batch_size)
        pred_binary = torch.round(pred_score)
        sum_correct += (pred_binary == Y_labels).sum().item()
      
        # Append predictions to lists
        Y_pred_score += pred_score.cpu().detach().numpy().tolist()
        Y_pred_binary += pred_binary.cpu().detach().numpy().tolist()
        Y_gr_truth += Y_labels.cpu().detach().numpy().tolist()
      
        # Log train step
        if mode == 'Train':
            logger.info(f'Epoch {epoch + 1} of {args.n_epochs}' +
                        f' Step {step_idx + 1:,} of {len(data_loader):,}')
        
    # Compute metrics
    avg_loss = sum_loss / total_entries
    avg_acc = sum_correct / total_entries
    
    # Print & log results
    print(f'\n{mode} loss: {avg_loss:.4f} and accuracy: {avg_acc:.4f}')
    logger.info(f'{mode} loss: {avg_loss:.4f} and accuracy: {avg_acc:.4f}')
    
    return avg_loss, avg_acc, Y_pred_score, Y_pred_binary, Y_gr_truth

#%% Main
def main():
    # Arg parsing
    args = utils.parse_args_f()
    
    # Path initialization train-dev
    path_train_dataset = os.path.join(args.input_dir, 'model_train.pkl')
    path_dev_dataset = os.path.join(args.input_dir, 'model_dev.pkl')
    path_test_dataset = os.path.join(args.input_dir, args.test_file)
    
    # Create ouput dir if not existing
    utils.make_dir_f(args.output_dir)
    
    # Instantiate logger
    logger = utils.get_logger_f(args.output_dir)
      
    # Global and seed initialization
    random.seed = args.seed
    _ = torch.manual_seed(args.seed)

    # Generate dataloaders
    if args.task == 'Train':
        # Load datasets
        print('Loading data')
        train_dataset = pd.read_pickle(path_train_dataset)
        dev_dataset = pd.read_pickle(path_dev_dataset)
        print('Done')
        # Convert to toy data if required
        if eval(args.train_toy_data) == True:
            train_dataset = train_dataset[0:args.len_train_toy_data]
            dev_dataset = dev_dataset[0:args.len_train_toy_data]
        # Instantiate dataclasses
        train_dataset = ECHR2_dataset(train_dataset)
        dev_dataset = ECHR2_dataset(dev_dataset)
        # Instantiate dataloaders
        train_dl = DataLoader(train_dataset, batch_size = args.batch_size_train,
                              shuffle = eval(args.shuffle_train),
                              drop_last = eval(args.drop_last_train))
        dev_dl = DataLoader(dev_dataset,
                            batch_size = int(args.batch_size_train * args.dev_train_ratio),
                            shuffle = False)

    elif args.task == 'Test':
        # Load datasets
        print('Loading data')
        test_dataset = pd.read_pickle(path_test_dataset)
        print('Done')
        # Instantiate dataclasses
        test_dataset = ECHR2_dataset(test_dataset)
        # Instantiate dataloaders
        test_dl = DataLoader(test_dataset,
                             batch_size = int(args.batch_size_test),
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

    # Train procedure
    if args.task == 'Train':
        # Save model parameters    
        utils.save_args_f(args)
        
        # Initializaztion
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        start_time = datetime.datetime.now()
        
        for epoch in tqdm(range(args.n_epochs), desc = 'Training dataset'):        
            # Train
            mode = 'Train'    
            train_loss, train_acc, _, _, _ = run_epoch_f(args, mode, model, criterion,
                                                         optimizer, logger, train_dl,                                              
                                                         device, epoch)
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
    
            # Validate
            mode = 'Validation'
            val_loss, val_acc, _, _, _ = run_epoch_f(args, mode, model, criterion,
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
        
        # Save train results
        utils.save_train_results_f(args, train_loss_history, train_acc_history,
                                   val_loss_history, val_acc_history, start_time)

    # Test procedure
    elif args.task == 'Test':
        mode = 'Test'
        # Load model
        path_model = os.path.join(args.output_dir, args.model_file)
        model.load_state_dict(torch.load(path_model)['model_state_dict'])
        # Compute predictions
        _, _, Y_pred_score, Y_pred_binary, Y_gr_truth = run_epoch_f(args, mode,
                                                                    model, criterion,
                                                                    optimizer, logger,
                                                                    test_dl, device,
                                                                    epoch = None)
        # Save test results
        utils.save_test_results_f(args, Y_pred_score, Y_pred_binary, Y_gr_truth)

if __name__ == "__main__":
    main()
