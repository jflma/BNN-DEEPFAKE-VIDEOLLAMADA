import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from models.face_processor import VideoFeatureExtractor, get_augmentations

class VideoDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, input_size=224, max_samples=None):
        self.real_images = self._get_image_paths(real_dir)
        self.fake_images = self._get_image_paths(fake_dir)
        
        # Limitar muestras si se especifica (para pruebas)
        if max_samples:
            self.real_images = self.real_images[:max_samples]
            self.fake_images = self.fake_images[:max_samples]
        
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform
        self.feature_extractor = VideoFeatureExtractor(input_size=input_size)
        
        print(f"Dataset cargado: {len(self.real_images)} reales, {len(self.fake_images)} falsos")
        
    def _get_image_paths(self, directory):
        """Obtiene todas las rutas de imágenes en el directorio"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
                    
        return image_paths
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = cv2.imread(img_path)
            
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {img_path}")
                
            # Extraer características
            features = self.feature_extractor.extract_features(image)
            
            label = self.labels[idx]
            
            # Aplicar transformaciones si están definidas
            if self.transform:
                # Convertir a formato HWC para albumentations
                features_transformed = self.transform(image=features.transpose(1, 2, 0))
                features = features_transformed['image']
            else:
                features = torch.FloatTensor(features)
                
            return features, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            # Retornar tensor de ceros en caso de error
            return torch.zeros((5, 224, 224), dtype=torch.float32), torch.tensor(0, dtype=torch.float32)

def get_data_loaders(real_dir, fake_dir, batch_size=16, train_ratio=0.8, input_size=224, max_samples=None):
    """Crea DataLoaders para entrenamiento y validación"""
    
    # Transformaciones para entrenamiento
    train_transform = get_augmentations()
    val_transform = None  # Sin aumentos para validación
    
    # Dataset completo
    full_dataset = VideoDeepfakeDataset(real_dir, fake_dir, transform=None, 
                                      input_size=input_size, max_samples=max_samples)
    
    # Dividir dataset
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Para reproducibilidad
    )
    
    # Aplicar transformaciones diferentes
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Entrenamiento: {len(train_dataset)} muestras")
    print(f"Validación: {len(val_dataset)} muestras")
    
    return train_loader, val_loader