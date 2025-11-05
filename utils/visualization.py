import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd

def plot_confusion_matrix(model, data_loader, device, save_path="results/figures/confusion_matrix.png"):
    """Genera y guarda matriz de confusión"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            preds = (torch.sigmoid(output) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Matriz de Confusión - Detección de Deepfakes')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_roc_curve(model, data_loader, device, save_path="results/figures/roc_curve.png"):
    """Genera y guarda curva ROC"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            probs = torch.sigmoid(output)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Detección de Deepfakes')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(model, data_loader, device, save_path="results/figures/precision_recall_curve.png"):
    """Genera y guarda curva Precision-Recall"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            probs = torch.sigmoid(output)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'Precision-Recall (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall - Detección de Deepfakes')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_precision

def generate_results_table(model, data_loader, device, save_path="results/tables/final_results.tex"):
    """Genera tabla LaTeX con resultados finales"""
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).int()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_targets, all_probs)
    
    # Crear tabla LaTeX
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Valor}} & \\textbf{{Unidad}} \\\\
\\hline
Precisión (Accuracy) & {accuracy:.4f} & - \\\\
Precision & {precision:.4f} & - \\\\
Recall & {recall:.4f} & - \\\\
F1-Score & {f1:.4f} & - \\\\
AUC & {auc_score:.4f} & - \\\\
\\hline
\\end{{tabular}}
\\caption{{Resultados del modelo de detección de deepfakes en el dataset FaceForensics++}}
\\label{{tab:resultados}}
\\end{{table}}
"""
    
    # Guardar tabla
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"Tabla LaTeX guardada en: {save_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score
    }