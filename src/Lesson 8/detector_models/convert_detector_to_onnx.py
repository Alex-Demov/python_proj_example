from ultralytics import YOLO
import os

# Путь к директории с моделями YOLO
model_dir = r"C:\USERS\ALEXEY\DESKTOP\LESSON 8\DETECTOR_MODELS"

# Список моделей для конвертации
models_to_convert = [
    ("yolov8n", os.path.join(model_dir, "yolov8n", "best.pt")),
    ("yolov10n", os.path.join(model_dir, "yolov10n", "best.pt")),
    ("yolo11n", os.path.join(model_dir, "yolo11n", "best.pt")),
]

# Конвертация моделей
for model_name, model_path in models_to_convert:
    try:
        model = YOLO(model_path)
        model.export(format="onnx")
        # Переименование файла для соответствия имени модели
        onnx_file_name = model_path.replace(".pt", ".onnx")
        new_onnx_file_name = os.path.join(model_dir, model_name, f"{model_name}.onnx")
        os.replace(onnx_file_name, new_onnx_file_name)
        print(f"Модель {model_name} успешно конвертирована в ONNX: {new_onnx_file_name}")

    except Exception as e:
        print(f"Ошибка при конвертации модели {model_name}: {e}")