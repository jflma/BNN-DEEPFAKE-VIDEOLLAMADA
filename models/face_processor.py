import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.feature import local_binary_pattern
import mediapipe as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FaceProcessor:
    def __init__(self, input_size=224):
        self.input_size = input_size
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        
    def detect_face(self, image):
        """Detecta y extrae el rostro principal"""
        try:
            # Convertir BGR a RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]
                
                # Coordenadas del bounding box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Añadir margen (20%)
                margin_x = int(width * 0.2)
                margin_y = int(height * 0.2)
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                width = min(w - x, width + 2 * margin_x)
                height = min(h - y, height + 2 * margin_y)
                
                # Extraer y redimensionar rostro
                face = image[y:y+height, x:x+width]
                return cv2.resize(face, (self.input_size, self.input_size))
            
        except Exception as e:
            print(f"Error en detección facial: {e}")
        
        # Fallback: redimensionar imagen completa
        return cv2.resize(image, (self.input_size, self.input_size))

class VideoFeatureExtractor:
    def __init__(self, input_size=224):
        self.input_size = input_size
        self.face_processor = FaceProcessor(input_size)
        
    def compute_fft_magnitude(self, image):
        """Calcula magnitud FFT para análisis de frecuencia"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        gray = cv2.resize(gray, (self.input_size, self.input_size))
        fft = fft2(gray.astype(float))
        fft_shift = fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        return cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    def compute_lbp(self, image, radius=3, n_points=24):
        """Calcula Local Binary Patterns para texturas faciales"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        gray = cv2.resize(gray, (self.input_size, self.input_size))
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        return cv2.normalize(lbp, None, 0, 1, cv2.NORM_MINMAX)
    
    def extract_features(self, image):
        """Extrae características multimodales para deepfake detection"""
        # Detectar y procesar rostro
        face = self.face_processor.detect_face(image)
        
        # Convertir a RGB si es necesario
        if len(face.shape) == 3 and face.shape[2] == 3:
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            rgb_face = face
            
        # Normalizar RGB
        rgb_normalized = rgb_face.astype(np.float32) / 255.0
        
        # Extraer características
        fft_magnitude = self.compute_fft_magnitude(face)
        lbp_texture = self.compute_lbp(face)
        
        # Expandir dimensiones para concatenación
        fft_expanded = np.expand_dims(fft_magnitude, axis=-1)
        lbp_expanded = np.expand_dims(lbp_texture, axis=-1)
        
        # Concatenar características (5 canales)
        features = np.concatenate([rgb_normalized, fft_expanded, lbp_expanded], axis=-1)
        
        return np.transpose(features, (2, 0, 1))  # (C, H, W)

def get_augmentations():
    """Configura aumentos de datos para entrenamiento"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ToFloat(),
        ToTensorV2(),
    ])