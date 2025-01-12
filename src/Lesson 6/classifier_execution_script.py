import argparse
from typing import Any
import os
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# Константы для классов
CLASS_NAMES = ['aircraft', 'ship']

def show_image_results(class_name: str, image: Any) -> None:
    """Отображает изображение с предварительно предсказанным классом."""
    cv2.putText(image, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Predicted Class", image)

def inference_classifier(classifier: Any, path_to_image: str) -> str:
    """Метод для инференса классификатора на единичном изображении."""
    # Преобразование изображения
    img = Image.open(path_to_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Добавляем размер батча
    with torch.no_grad():
        classifier.eval()
        output = classifier(img_tensor)
        _, predicted = torch.max(output, 1)
        
    return CLASS_NAMES[predicted.item()]

def load_classifier(name_of_classifier: str, path_to_pth_weights: str, device: str) -> Any:
    """Метод для загрузки классификатора."""
    if name_of_classifier == "resnet18":
        model = models.resnet18(weights=None)
    elif name_of_classifier == "efficientnet":
        model = models.efficientnet_b0(weights=None)
    elif name_of_classifier == "regnet":
        model = models.regnet_x_400mf(weights=None)
    else:
        raise ValueError(f"Unknown classifier: {name_of_classifier}")

    num_ftrs = model.fc.in_features if hasattr(model, 'fc') else model.classifier[1].in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(path_to_pth_weights, map_location=device))
    model.to(device)
    return model

def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов."""
    parser = argparse.ArgumentParser(description="Скрипт для выполнения классификатора на единичном изображении или папке с изображениями")
    parser.add_argument("--name_of_classifier", "-nc", type=str, help="Название классификатора")
    parser.add_argument("--path_to_weights", "-wp", type=str, help="Путь к PTH-файлу с весами классификатора")
    parser.add_argument("--path_to_content", "-cp", type=str, help="Путь к одиночному изображению/папке с изображениями")
    parser.add_argument("--use_cuda", "-uc", action="store_true", help="Использовать ли CUDA для инференса")
    args = parser.parse_args()
    return args

def main() -> None:
    """Основная логика работы с классификатором."""
    args = arguments_parser()
    name_of_classifier = args.name_of_classifier
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    print(f"Name of classifier: {name_of_classifier}")
    print(f"Path to content: {path_to_content}")
    print(f"Path to weights: {path_to_weights}")
    print(f"Device: {device}")

    classifier = load_classifier(name_of_classifier, path_to_weights, device)

    if os.path.isfile(path_to_content):
        class_name = inference_classifier(classifier, path_to_content)
        image = cv2.imread(path_to_content)
        show_image_results(class_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif os.path.isdir(path_to_content):
        images = [f for f in os.listdir(path_to_content) if f.endswith(('png', 'jpg', 'jpeg'))]
        if not images:
            print("В папке не найдено изображений.")
            return
            
        current_index = 0

        while True:
            img_path = os.path.join(path_to_content, images[current_index])
            class_name = inference_classifier(classifier, img_path)
            image = cv2.imread(img_path)
            show_image_results(class_name, image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Выход
                break
            elif key == ord('a'):  # "a" для стрелки влево
                current_index = (current_index - 1) % len(images)  # Циклический переход влево
            elif key == ord('d'):  # "d" для стрелки вправо
                current_index = (current_index + 1) % len(images)  # Циклический переход вправо
            
            # Показать следующее изображение
            img_path = os.path.join(path_to_content, images[current_index])
            class_name = inference_classifier(classifier, img_path)
            image = cv2.imread(img_path)
            show_image_results(class_name, image)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




