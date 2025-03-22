import torch
import torchvision.models as models
import os
from collections import OrderedDict

def convert_to_onnx(model_path, model_name, num_classes=2, input_shape=(1, 3, 224, 224)):
    """
    Конвертирует модель PyTorch в формат ONNX.

    Args:
        model_path (str): Путь к файлу модели PyTorch.
        model_name (str): Имя модели (используется для имени файла ONNX).
        num_classes (int): Количество классов в модели.
        input_shape (tuple): Форма входных данных для модели.
    """
    try:
        # Загрузка модели PyTorch
        if "EfficientNet" in model_name:
            model = models.efficientnet_b0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        elif "RegNet" in model_name:
            model = models.regnet_x_400mf(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif "ResNet" in model_name:
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Проверка и обработка state_dict, если используется DataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # удаляем 'module.'
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        # Создание фиктивного входного тензора
        dummy_input = torch.randn(input_shape)

        # Экспорт модели в формат ONNX
        onnx_path = os.path.join(os.path.dirname(model_path), f"{model_name}.onnx")
        torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

        print(f"Модель {model_name} успешно конвертирована в ONNX: {onnx_path}")

    except Exception as e:
        print(f"Ошибка при конвертации модели {model_name}: {e}")

# Путь к директории с моделями
model_dir = r"C:\USERS\ALEXEY\DESKTOP\LESSON 8\BEST_MODEL_TORCH"

# Список моделей для конвертации
models_to_convert = [
    ("best_model_EfficientNet.pth", "EfficientNet"),
    ("best_model_RegNet.pth", "RegNet"),
    ("best_model_ResNet.pth", "ResNet"),
]

# Конвертация моделей
for model_file, model_name in models_to_convert:
    model_path = os.path.join(model_dir, model_file)
    convert_to_onnx(model_path, model_name, num_classes=2)