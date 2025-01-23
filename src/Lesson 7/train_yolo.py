from ultralytics import YOLO

# Параметры для обучения
imgsz = 640
epochs = 50
batch = 8
num_workers = 4

# Спецификации детекторов
models = ['yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt']

for model_name in models:
    print(f"Начинаем обучение {model_name}...")
    model = YOLO(model_name)  # Загружаем модель
    model.train(data='dataset.yaml',
                imgsz=imgsz,
                epochs=epochs,
                batch=batch,
                workers=num_workers)
    print(f"Обучение {model_name} завершено!")