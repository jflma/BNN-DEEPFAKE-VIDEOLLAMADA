import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DeepfakeTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizer y loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs']
        )
        
        # Métricas
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.auc_scores = []
        
        # Mejor modelo
        self.best_auc = 0
        self.best_model_path = "results/checkpoints/best_model.pth"
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Entrenamiento")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data).squeeze()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        return running_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validación"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data).squeeze()
                loss = self.criterion(output, target)
                val_loss += loss.item()
                
                probs = torch.sigmoid(output).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        auc = roc_auc_score(all_targets, all_probs)
        
        return val_loss / len(self.val_loader), accuracy, precision, recall, f1, auc
    
    def train(self, epochs):
        print("Iniciando entrenamiento del detector de deepfakes...")
        print(f"Dispositivo: {self.device}")
        print(f"Parámetros del modelo: {self.model.get_parameter_count():,}")
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f'Época {epoch+1}/{epochs}')
            print(f"{'='*50}")
            
            # Entrenamiento
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validación
            val_loss, accuracy, precision, recall, f1, auc = self.validate()
            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1_scores.append(f1)
            self.auc_scores.append(auc)
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Imprimir métricas
            print(f"Pérdida Entrenamiento: {train_loss:.4f}")
            print(f"Pérdida Validación:   {val_loss:.4f}")
            print(f"Precisión:            {accuracy:.4f}")
            print(f"Precision:            {precision:.4f}")
            print(f"Recall:               {recall:.4f}")
            print(f"F1-Score:             {f1:.4f}")
            print(f"AUC:                  {auc:.4f}")
            
            # Guardar mejor modelo
            if auc > self.best_auc:
                self.best_auc = auc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"¡Nuevo mejor modelo guardado! AUC: {auc:.4f}")
    
    def plot_training_curves(self, save_path="results/figures/training_curves.png"):
        """Genera gráficas del entrenamiento"""
        plt.figure(figsize=(15, 10))
        
        # Gráfica de pérdidas
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Entrenamiento', linewidth=2)
        plt.plot(self.val_losses, label='Validación', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Pérdida durante el Entrenamiento')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfica de precisión
        plt.subplot(2, 3, 2)
        plt.plot(self.accuracies, linewidth=2, color='green')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.title('Precisión en Validación')
        plt.grid(True, alpha=0.3)
        
        # Gráfica de AUC
        plt.subplot(2, 3, 3)
        plt.plot(self.auc_scores, linewidth=2, color='purple')
        plt.xlabel('Época')
        plt.ylabel('AUC')
        plt.title('AUC en Validación')
        plt.grid(True, alpha=0.3)
        
        # Gráfica de F1-Score
        plt.subplot(2, 3, 4)
        plt.plot(self.f1_scores, linewidth=2, color='red')
        plt.xlabel('Época')
        plt.ylabel('F1-Score')
        plt.title('F1-Score en Validación')
        plt.grid(True, alpha=0.3)
        
        # Gráfica de Precision y Recall
        plt.subplot(2, 3, 5)
        plt.plot(self.precisions, label='Precision', linewidth=2)
        plt.plot(self.recalls, label='Recall', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Métrica')
        plt.title('Precision y Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Métricas finales
        plt.subplot(2, 3, 6)
        final_metrics = [self.accuracies[-1], self.precisions[-1], 
                        self.recalls[-1], self.f1_scores[-1], self.auc_scores[-1]]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        bars = plt.bar(metric_names, final_metrics, color=colors, alpha=0.7)
        plt.ylabel('Valor')
        plt.title('Métricas Finales')
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, final_metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráficas guardadas en: {save_path}")