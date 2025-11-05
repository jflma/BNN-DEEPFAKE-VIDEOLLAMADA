import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_frames_from_videos(video_dir, output_dir, frames_per_video=50, target_size=(224, 224)):
    """Extrae frames de videos para crear dataset de imágenes"""
    
    if not os.path.exists(video_dir):
        print(f"Directorio de videos no encontrado: {video_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No se encontraron videos en: {video_dir}")
        return
    
    print(f"Extrayendo frames de {len(video_files)} videos...")
    
    for video_file in tqdm(video_files, desc="Procesando videos"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Crear subdirectorio para este video
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"Video vacío o corrupto: {video_file}")
                continue
            
            # Calcular intervalos para frames equidistantes
            frame_interval = max(1, total_frames // frames_per_video)
            saved_frames = 0
            
            for frame_idx in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and saved_frames < frames_per_video:
                    # Redimensionar frame
                    frame_resized = cv2.resize(frame, target_size)
                    
                    # Guardar frame
                    frame_path = os.path.join(video_output_dir, f"frame_{saved_frames:04d}.jpg")
                    cv2.imwrite(frame_path, frame_resized)
                    saved_frames += 1
                else:
                    break
            
            cap.release()
            
        except Exception as e:
            print(f"Error procesando video {video_file}: {e}")
            continue

def prepare_faceforensics_data():
    """Prepara los datos de FaceForensics++ extrayendo frames"""
    
    # Rutas de entrada (videos)
    real_videos_dir = "data/raw/FaceForensics/original_sequences/youtube/raw/videos"
    fake_videos_dir = "data/raw/FaceForensics/manipulated_sequences/DeepFake/raw/videos"
    
    # Rutas de salida (frames)
    real_frames_dir = "data/processed/real"
    fake_frames_dir = "data/processed/fake"
    
    print("Extrayendo frames de videos reales...")
    extract_frames_from_videos(real_videos_dir, real_frames_dir, frames_per_video=50)
    
    print("Extrayendo frames de videos falsos...")
    extract_frames_from_videos(fake_videos_dir, fake_frames_dir, frames_per_video=50)
    
    print("¡Extracción de frames completada!")

if __name__ == "__main__":
    prepare_faceforensics_data()