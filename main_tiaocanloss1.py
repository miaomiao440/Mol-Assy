import os
import sys
import time
import json
import random
import itertools
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.set_loglevel("error")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import gc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from models.loss import binary_cross_entropy
from MPNN import SD
from loaddataset_new2 import GraphDataset4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

def run_grid_search(n_trials=1, num_repeats=1):  
    #======================================================
    result_dir = "./result"###########################################################
    os.makedirs(result_dir, exist_ok=True)

    def is_same_config(cfg1, cfg2):
        return json.dumps(cfg1, sort_keys=True) == json.dumps(cfg2, sort_keys=True)

    def is_valid_structure(trial):
        return len(trial['n_out_feats']) == len(trial['n_heads'])

    completed_trials = []

    param_grid = {
        'epochs': [1],
        'lr': [0.001],
        'batch_size': [128],
        'dropout': [0.15],
        'dropedge_rate': [0.3],
        'weight_decay': [0],
        'in_node_features': [79],
        'in_edge_features': [10],
        'edge_feature': [128],
        'hidd_dim': [128],
        'n_out_feats': [   
        [64,128,128]
        ],
        'n_heads': [
            [2,2,2]
        ],
        'attn_heads': [[4]],
        'attn_dim_head': [64],
        'T': [7],
        'nclass': [1],
        'seed':[42]
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    valid_combinations = []
    for p in all_combinations:
        if is_valid_structure(p):
            valid_combinations.append(p)
    filtered_combinations = []
    for p in valid_combinations:
        is_duplicate = False
        for d in completed_trials:
            if is_same_config(p, d):
                is_duplicate = True
                break  
        if not is_duplicate:
            filtered_combinations.append(p)
    
    random.seed(42)
    param_combinations = random.sample(filtered_combinations, min(n_trials, len(filtered_combinations)))
    print("Loading dataset...")
    full_dataset = GraphDataset4("./data/train_val4_zs.xlsx", bidrection=False)

    all_results = []  
    summary_results = []  
    detailed_file = os.path.join(result_dir, "jieguo.xlsx")
    summary_file = os.path.join(result_dir, "all.xlsx")

    for trial_idx, trial in enumerate(param_combinations):
        trial_dir = os.path.join(result_dir, f"canshu_{trial_idx+1}")
        os.makedirs(trial_dir, exist_ok=True)
        params_file = os.path.join(trial_dir, "parameters.txt")
        with open(params_file, 'w') as f:
            json.dump(trial, f, indent=4)

        trial_repeat_metrics = []
        
        for repeat_idx in range(num_repeats):  
            print(f"\n=== parameter {trial_idx+1}/{len(param_combinations)}-repeat {repeat_idx+1}/{num_repeats} ===")
            current_seed = 42  
            random.seed(current_seed)
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

            kf = KFold(n_splits=5, shuffle=True, random_state=current_seed)

            fold_metrics = []
            train_metrics_all = []  
            val_metrics_all = []    

            for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
                print(f"\n---{fold+1}/5fold---")
                
                train_dataset = Subset(full_dataset, train_idx)
                val_dataset = Subset(full_dataset, val_idx)
                train_loader = DataLoader(train_dataset,
                                          batch_size=trial['batch_size'],
                                          shuffle=True)

                val_loader = DataLoader(val_dataset,
                                        batch_size=trial['batch_size'])

                model = SD(
                    nclass=trial['nclass'],
                    in_node_features=trial['in_node_features'],
                    in_edge_features=trial['in_edge_features'],
                    hidd_dim=trial['hidd_dim'],
                    n_out_feats=trial['n_out_feats'],
                    n_heads=trial['n_heads'],
                    edge_feature=trial['edge_feature'],
                    dp=trial['dropout'],
                    dropedge_rate=trial['dropedge_rate'],
                    T=trial['T']
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=trial['lr'],
                                             weight_decay=trial['weight_decay'])
                
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.8, patience=5, verbose=False, 
                    threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-7
                )
                criterion = binary_cross_entropy

                best_f1 = 0
                patience, no_improve = 20, 0
                
                train_losses = []
                val_losses = []
                best_model_state = None
                best_epoch = 0
                
                train_f1, train_acc, train_auc, train_auc_pr, train_precision, train_recall = 0, 0, 0, 0, 0, 0

                for epoch in range(trial['epochs']):
                    model.train()
                    train_loss = []
                    y_true_train, y_prob_train = [], []

                    for batch in train_loader:
                        data1, data2, edge1, edge2, weight_D2S, weight_S2D, sample_ids = batch
                        data1, data2 = data1.to(device), data2.to(device)
                        edge1, edge2 = edge1.to(device), edge2.to(device)
                        weight = weight_D2S.to(device)
                        target = data1.y.to(device)
                        out = model(data1, data2, edge1, edge2)
                        out = torch.sigmoid(out)
                        loss = criterion(out, target, weight, True)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss.append(loss.item())

                    avg_train_loss = np.mean(train_loss)
                    train_losses.append(avg_train_loss)

                    model.eval()
                    y_true, y_pred, y_prob = [], [], []
                    val_loss = []
                    all_val_sample_ids = [] 

                    with torch.no_grad():
                        for batch in val_loader:
                            data1, data2, edge1, edge2, weight_D2S, weight_S2D, sample_ids = batch
                            data1, data2 = data1.to(device), data2.to(device)
                            edge1, edge2 = edge1.to(device), edge2.to(device)
                            weight = weight_D2S.to(device)
                            target = data1.y.to(device)
                            out = model(data1, data2, edge1, edge2)
                            prob = torch.sigmoid(out)
                            
                            loss_val = criterion(prob, target, weight, True)
                            val_loss.append(loss_val.item())
                            pred = (prob >= 0.5).float()
                            y_true.extend(target.cpu().numpy())
                            y_pred.extend(pred.cpu().numpy())
                            y_prob.extend(prob.cpu().numpy())
                            all_val_sample_ids.extend(sample_ids)  
                    avg_val_loss = np.mean(val_loss)
                    val_losses.append(avg_val_loss)
                    scheduler.step(avg_val_loss)

                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)
                    y_prob = np.array(y_prob)

                    if len(np.unique(y_true)) < 2:
                        f1, auc, acc, precision, recall, auc_pr = 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                    else:
                        f1 = f1_score(y_true, y_pred)
                        auc = roc_auc_score(y_true, y_prob)
                        acc = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        auc_pr = average_precision_score(y_true, y_prob)

                    if f1 > best_f1:
                        best_f1 = f1
                        no_improve = 0
                        best_model_state = model.state_dict().copy()
                        best_epoch = epoch
                        
                        val_df = pd.DataFrame({
                            'sample_id': all_val_sample_ids,
                            'true_label': y_true,
                            'pred_prob': y_prob,
                            'pred_label': y_pred
                        })
                        
                        val_pred_file = os.path.join(trial_dir, f"val_predictions_repeat{repeat_idx+1}_fold{fold+1}.xlsx")
                        val_df.to_excel(val_pred_file, index=False)
                        
                        model.eval()
                        train_true, train_prob, train_pred = [], [], []
                        all_train_sample_ids = []  
                        
                        with torch.no_grad():
                            for batch in train_loader:
                                data1, data2, edge1, edge2, weight_D2S, weight_S2D, sample_ids = batch
                                data1, data2 = data1.to(device), data2.to(device)
                                edge1, edge2 = edge1.to(device), edge2.to(device)
                                
                                out_train = model(data1, data2, edge1, edge2)
                                prob_train = torch.sigmoid(out_train)
                                pred_train = (prob_train >= 0.5).float()
                                
                                train_true.extend(data1.y.cpu().numpy())
                                train_prob.extend(prob_train.cpu().numpy())
                                train_pred.extend(pred_train.cpu().numpy())
                                all_train_sample_ids.extend(sample_ids)                          
                        train_true = np.array(train_true)
                        train_prob = np.array(train_prob)
                        train_pred = np.array(train_pred)
                        if len(np.unique(train_true)) >= 2:
                            train_f1 = f1_score(train_true, train_pred)
                            train_auc = roc_auc_score(train_true, train_prob)
                            train_acc = accuracy_score(train_true, train_pred)
                            train_precision = precision_score(train_true, train_pred, zero_division=0)
                            train_recall = recall_score(train_true, train_pred, zero_division=0)
                            train_auc_pr = average_precision_score(train_true, train_prob)
                        else:
                            train_f1, train_auc, train_acc, train_precision, train_recall, train_auc_pr = 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        
                        train_df = pd.DataFrame({
                            'sample_id': all_train_sample_ids,
                            'true_label': train_true,
                            'pred_prob': train_prob,
                            'pred_label': train_pred
                        })
                        
                        train_pred_file = os.path.join(trial_dir, f"train_predictions_repeat{repeat_idx+1}_fold{fold+1}.xlsx")
                        train_df.to_excel(train_pred_file, index=False)

                    else:
                        no_improve += 1

                    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, F1={train_f1:.4f},ACC={train_acc:.4f},PRE={train_precision:.4f},recall={train_recall:.4f}, AUC={train_auc:.4f}, PR={train_auc_pr:.4f}")
                    print(f"Epoch {epoch+1}:  Val Loss={avg_val_loss:.4f}, F1={f1:.4f},ACC={acc:.4f},PRE={precision:.4f},recall={recall:.4f}, AUC={auc:.4f}, PR={auc_pr:.4f}")

                    if no_improve >= patience:
                        print("Early stopping triggered!")
                        break
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Val Loss')
                plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Loss - Repeat {repeat_idx+1} Fold {fold+1}')
                plt.legend()
                loss_plot_file = os.path.join(trial_dir, f"loss_repeat{repeat_idx+1}_fold{fold+1}.png")
                plt.savefig(loss_plot_file)
                plt.close()
                
                train_loss_file = os.path.join(trial_dir, f"train_loss_repeat{repeat_idx+1}_fold{fold+1}.txt")
                val_loss_file = os.path.join(trial_dir, f"val_loss_repeat{repeat_idx+1}_fold{fold+1}.txt")
                
                with open(train_loss_file, 'w') as f:
                    for loss in train_losses:
                        f.write(f"{loss}\n")
                
                with open(val_loss_file, 'w') as f:
                    for loss in val_losses:
                        f.write(f"{loss}\n")
                
                best_metrics_file = os.path.join(trial_dir, f"best_metrics_repeat{repeat_idx+1}_fold{fold+1}.txt")
                with open(best_metrics_file, 'w') as f:
                    f.write(f"Repeat: {repeat_idx+1}, Fold: {fold+1}\n")
                    f.write(f"Best Epoch: {best_epoch+1}\n")
                    f.write(f"Training Metrics:\n")
                    f.write(f"  F1: {train_f1:.4f}\n")
                    f.write(f"  Accuracy: {train_acc:.4f}\n")
                    f.write(f"  AUC-ROC: {train_auc:.4f}\n")
                    f.write(f"  AUC-PR: {train_auc_pr:.4f}\n")
                    f.write(f"  Precision: {train_precision:.4f}\n")
                    f.write(f"  Recall: {train_recall:.4f}\n")
                    f.write(f"Validation Metrics:\n")
                    f.write(f"  F1: {best_f1:.4f}\n")
                    f.write(f"  Accuracy: {acc:.4f}\n")
                    f.write(f"  AUC-ROC: {auc:.4f}\n")
                    f.write(f"  AUC-PR: {auc_pr:.4f}\n")
                    f.write(f"  Precision: {precision:.4f}\n")
                    f.write(f"  Recall: {recall:.4f}\n")

                fold_metrics.append(best_f1)
                
                train_metrics_all.append({
                    'f1': train_f1, 'acc': train_acc, 'auc_roc': train_auc,
                    'auc_pr': train_auc_pr, 'pre': train_precision, 'recall': train_recall
                })
                val_metrics_all.append({
                    'f1': best_f1, 'acc': acc, 'auc_roc': auc,
                    'auc_pr': auc_pr, 'pre': precision, 'recall': recall
                })
                
                all_results.append({
                    'num.': trial_idx + 1,
                    'repeat': repeat_idx + 1,
                    'fold': fold + 1,
                    'data': 'train',
                    'f1': train_f1,
                    'acc': train_acc,
                    'auc_roc': train_auc,
                    'auc_pr': train_auc_pr,
                    'pre': train_precision,
                    'recall': train_recall
                })
                
                all_results.append({
                    'num.': trial_idx + 1,
                    'repeat': repeat_idx + 1,
                    'fold': fold + 1,
                    'data': 'val',
                    'f1': best_f1,
                    'acc': acc,
                    'auc_roc': auc,
                    'auc_pr': auc_pr,
                    'pre': precision,
                    'recall': recall
                })
                
                detailed_df = pd.DataFrame(all_results)
                detailed_df.to_excel(detailed_file, index=False)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            train_metrics_mean = {key: np.mean([m[key] for m in train_metrics_all]) for key in train_metrics_all[0].keys()}
            train_metrics_std = {key: np.std([m[key] for m in train_metrics_all]) for key in train_metrics_all[0].keys()}
            
            val_metrics_mean = {key: np.mean([m[key] for m in val_metrics_all]) for key in val_metrics_all[0].keys()}
            val_metrics_std = {key: np.std([m[key] for m in val_metrics_all]) for key in val_metrics_all[0].keys()}
            
            trial_repeat_metrics.append({
                'repeat': repeat_idx + 1,
                'train_metrics': train_metrics_mean,
                'val_metrics': val_metrics_mean,
                'train_std': train_metrics_std,
                'val_std': val_metrics_std
            })
            
            summary_results.append({
                'num.': trial_idx + 1,
                'repeat': repeat_idx + 1,
                'data': 'train',
                'f1_mean': train_metrics_mean['f1'],
                'acc_mean': train_metrics_mean['acc'],
                'auc_roc_mean': train_metrics_mean['auc_roc'],
                'auc_pr_mean': train_metrics_mean['auc_pr'],
                'pre_mean': train_metrics_mean['pre'],
                'recall_mean': train_metrics_mean['recall'],
                'f1_std': train_metrics_std['f1']
            })
            
            summary_results.append({
                'num.': trial_idx + 1,
                'repeat': repeat_idx + 1,
                'data': 'val',
                'f1_mean': val_metrics_mean['f1'],
                'acc_mean': val_metrics_mean['acc'],
                'auc_roc_mean': val_metrics_mean['auc_roc'],
                'auc_pr_mean': val_metrics_mean['auc_pr'],
                'pre_mean': val_metrics_mean['pre'],
                'recall_mean': val_metrics_mean['recall'],
                'f1_std': val_metrics_std['f1']
            })

            repeat_f1_mean = np.mean(fold_metrics)
            
            summary_df = pd.DataFrame(summary_results)
            summary_df.to_excel(summary_file, index=False)

def main():
    print("Starting grid search...")
    start_time = time.time()
    #======================================================
    run_grid_search(n_trials=1, num_repeats=2)  ########################################
    print(f"Grid search finished in {round((time.time() - start_time) / 60, 2)} minutes")

if __name__ == "__main__":
    main()