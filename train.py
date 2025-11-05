import os
import torch
import yaml
from models.bnn_model import BinaryDeepfakeDetector
from utils.data_loader import get_data_loaders
from training.trainer import DeepfakeTrainer
from utils.visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve,
    generate_results_table
)

def load_config():
    """Carga la configuración desde el archivo YAML"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    # Cargar configuración
    config = load_config()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Crear directorios de resultados
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    # Rutas del dataset
    real_dir = config['data']['real_dir']
    fake_dir = config['data']['fake_dir']
    
    # Verificar que el dataset existe
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("Error: No se encuentran los directorios del dataset procesado.")
        print("Por favor, ejecuta primero: python scripts/extract_frames.py")
        print("Asegúrate de tener los videos originales en data/raw/FaceForensics/")
        return
    
    # DataLoaders
    print("Cargando dataset de deepfakes...")
    train_loader, val_loader = get_data_loaders(
        real_dir=real_dir,
        fake_dir=fake_dir,
        batch_size=config['training']['batch_size'],
        train_ratio=config['training']['train_ratio'],
        input_size=config['model']['input_size'],
        max_samples=1000  # Limitar para pruebas, quitar para entrenamiento completo
    )
    
    # Modelo
    model = BinaryDeepfakeDetector(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        input_channels=5
    )
    
    # Entrenamiento
    trainer = DeepfakeTrainer(model, train_loader, val_loader, device, config)
    trainer.train(epochs=config['training']['epochs'])
    
    # Gráficas de entrenamiento
    trainer.plot_training_curves()
    
    # ===== EVALUACIÓN FINAL =====
    print("\n" + "="*60)
    print("EVALUACIÓN FINAL DEL MODELO")
    print("="*60)
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('results/checkpoints/best_model.pth'))
    model.to(device)
    
    # Generar todas las gráficas de evaluación
    print("Generando gráficas de evaluación...")
    
    # Matriz de confusión
    cm = plot_confusion_matrix(model, val_loader, device)
    
    # Curva ROC
    auc_score = plot_roc_curve(model, val_loader, device)
    
    # Curva Precision-Recall
    avg_precision = plot_precision_recall_curve(model, val_loader, device)
    
    # Tabla de resultados LaTeX
    final_metrics = generate_results_table(model, val_loader, device)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL - DETECCIÓN DE DEEPFAKES EN VIDEOCALLS")
    print("="*60)
    print(f"Accuracy:    {final_metrics['accuracy']:.4f}")
    print(f"Precision:   {final_metrics['precision']:.4f}")
    print(f"Recall:      {final_metrics['recall']:.4f}")
    print(f"F1-Score:    {final_metrics['f1_score']:.4f}")
    print(f"AUC:         {final_metrics['auc_score']:.4f}")
    print(f"Parámetros:  {model.get_parameter_count():,}")
    print("="*60)
    
    # Guardar métricas en archivo
    with open('results/final_metrics.txt', 'w') as f:
        f.write("MÉTRICAS FINALES - DETECTOR DE DEEPFAKES\n")
        f.write("="*50 + "\n")
        for metric, value in final_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"total_params: {model.get_parameter_count()}\n")
    
    print("\n¡Entrenamiento y evaluación completados!")
    print("Resultados guardados en la carpeta 'results/'")

if __name__ == "__main__":
    main()