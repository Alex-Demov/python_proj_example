import os
import xml.etree.ElementTree as ET
from PIL import Image

# Запрос адресов папок и XML-файла у пользователя
test_folder = input("Ввести путь к папке с тестовыми изображениями: ")
train_folder = input("Ввести путь к папке с обучающими изображениями: ")
output_test_folder = input("Ввести путь к папке для сохранения вырезанных тестовых изображений: ")
output_train_folder = input("Ввести путь к папке для сохранения вырезанных обучающих изображений: ")
annotations_file = input("Ввести путь к XML-файлу с аннотациями: ")

os.makedirs(output_test_folder, exist_ok=True)
os.makedirs(output_train_folder, exist_ok=True)

# Парсинг XML
tree = ET.parse(annotations_file)
root = tree.getroot()

for image in root.findall("image"):
    image_name = image.get("name")
    subset = image.get("subset")
    boxes = image.findall("box")

    # Определяем папку для сохранения
    output_folder = output_test_folder if subset == "Test" else output_train_folder

    # Загружаем изображение
    image_path = os.path.join(test_folder if subset == "Test" else train_folder, image_name)
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Изображение {image_name} не найдено.")
        continue

    for i, box in enumerate(boxes):
        label = box.get("label")
        xtl = int(float(box.get("xtl")))
        ytl = int(float(box.get("ytl")))
        xbr = int(float(box.get("xbr")))
        ybr = int(float(box.get("ybr")))

        # Вырезаем объект
        cropped_img = img.crop((xtl, ytl, xbr, ybr))

        # Формируем уникальное название для сохранившегося файла
        cropped_img_name = f"{label}_{os.path.splitext(image_name)[0]}_{i}.png"  # добавляем индекс к имени
        cropped_img_path = os.path.join(output_folder, cropped_img_name)

        # Сохраняем изображение
        cropped_img.save(cropped_img_path)

print("Обработка изображений завершена")
