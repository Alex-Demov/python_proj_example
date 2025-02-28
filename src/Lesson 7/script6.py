import argparse
import cv2
import torch
import os

from ultralytics import YOLO


def inference_detector(detector: YOLO, device: torch.device, path_to_image: str) -> None:
    """Запуск детектора на изображении и отображение результатов.

    Args:
        detector (YOLO): Загруженная модель YOLO.
        device (torch.device): Устройство для выполнения (CPU или CUDA).
        path_to_image (str): Путь к изображению.
    """
    detector.to(device)
    results = detector(path_to_image)
    img = results[0].plot()
    cv2.imshow('Detection Results', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def load_detector(path_to_pt_weights: str, use_cuda: bool) -> tuple[YOLO, torch.device]:
    """Загрузка модели YOLO.

    Args:
        path_to_pt_weights (str): Путь к весам модели.
        use_cuda (bool): Использовать ли CUDA.

    Returns:
        tuple[YOLO, torch.device]: Модель и устройство.
    """
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    return YOLO(path_to_pt_weights), device


def main():
    """Основная логика скрипта."""
    parser = argparse.ArgumentParser(description="Запуск детектора YOLO на изображении или папке.")
    parser.add_argument("--path_to_weights", "-wp", type=str, required=True, help="Путь к весам модели (.pt).")
    parser.add_argument("--path_to_content", "-cp", type=str, required=True, help="Путь к изображению или папке с изображениями.")
    parser.add_argument("--use_cuda", "-uc", action="store_true", help="Использовать CUDA, если доступно.")
    args = parser.parse_args()

    detector, device = load_detector(args.path_to_weights, args.use_cuda)

    if os.path.isfile(args.path_to_content):
        inference_detector(detector, device, args.path_to_content)
        cv2.waitKey(0)
    elif os.path.isdir(args.path_to_content):
        images = [os.path.join(args.path_to_content, img) for img in os.listdir(args.path_to_content) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            current_index = 0
            while True:
                inference_detector(detector, device, images[current_index])
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('a') and current_index > 0:
                    current_index -= 1
                elif key == ord('d') and current_index < len(images) - 1:
                    current_index += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()