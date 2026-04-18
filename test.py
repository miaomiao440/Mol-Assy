import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
from torch_geometric.loader import DataLoader
from MPNN import SD
from loaddataset_new2 import GraphDataset4
import random

def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    #model_path = Path("./3_17_1_0.pt")
    model_path = Path("./test/fold5_best.pth")
    test_data_path = "./data/all-c1_pairs.xlsx"
    config_path = "./test/parameters.txt"
    #config_path = "./3_17_1_0.json"
    
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    test_dataset = GraphDataset4(test_data_path, bidrection=False)
    
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    loader = DataLoader(
        test_dataset, 
        batch_size=64,
        shuffle=False,  
        generator=generator
    )

    with open(config_path, 'r') as f:
        model_params = json.load(f)

    model = SD(
        nclass=model_params['nclass'],
        in_node_features=model_params['in_node_features'],
        in_edge_features=model_params['in_edge_features'],
        hidd_dim=model_params['hidd_dim'],
        n_out_feats=model_params['n_out_feats'],
        n_heads=model_params['n_heads'],
        edge_feature=model_params['edge_feature'],
        dp=model_params['dropout'],
        T=model_params['T'],
        dropedge_rate=model_params['dropedge_rate']
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    print(f"Model is in training mode: {model.training}")
    
    for name, module in model.named_modules():
        if hasattr(module, 'dropedge_rate'):
            print(f"Found DropEdge layer: {name}, rate={module.dropedge_rate}, training={module.training}")

    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []  

    with torch.no_grad():
        for batch in loader:
            data1, data2, edge1, edge2, _, _ = batch
            data1, data2 = data1.to(device), data2.to(device)
            edge1, edge2 = edge1.to(device), edge2.to(device)
            
            out = model(data1, data2, edge1, edge2)
            prob = torch.sigmoid(out).cpu().numpy().flatten()  
            pred_labels = (prob > 0.5).astype(int)  
            
            label = data1.y.view(-1, 2)[:, 0].cpu().numpy()
            
            all_true_labels.extend(label)
            all_pred_probs.extend(prob)
            all_pred_labels.extend(pred_labels)
    for i in range(min(10, len(all_pred_probs))):
        print(f"sample {i}: true={all_true_labels[i]}, predicted probability={all_pred_probs[i]:.6f}, predicted lable={all_pred_labels[i]}")

    result_df = pd.DataFrame({
        'sample_idx': range(len(all_pred_probs)),
        'true_label': all_true_labels,
        'pred_prob': all_pred_probs,
        'pred_label': all_pred_labels
    })
    
    result_path = "predictions_sxc.csv"
    result_df.to_csv(result_path, index=False)
    print(f"\npath: {result_path}")


if __name__ == "__main__":
    main()