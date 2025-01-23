#заменить первый символ "1" на "0" в каждой строке текстового файла, для дальнейшего корректного запуска обучения yolo
import os

# Путь к директориям с файлами
train_directory = r'C:\Users\Alexey\Desktop\Lesson 7\project_annotations\obj_Train_data'
test_directory = r'C:\Users\Alexey\Desktop\Lesson 7\project_annotations\obj_Test_data'

# Обработка файлов в директории train
for i in range(40):
    file_name = f'boat{i}.txt'
    file_path = os.path.join(train_directory, file_name)

    # Проверка существования файла
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Замена первого числа на 0 в каждой строке
        modified_lines = [line.replace('1 ', '0 ', 1) for line in lines]

        # Сохранение изменений в файл
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f'Файл {file_name} из train успешно обработан!')

# Обработка файлов в директории test
for i in range(40, 50):
    file_name = f'boat{i}.txt'
    file_path = os.path.join(test_directory, file_name)

    # Проверка существования файла
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Замена первого числа на 0 в каждой строке
        modified_lines = [line.replace('1 ', '0 ', 1) for line in lines]

        # Сохранение изменений в файл
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f'Файл {file_name} из test успешно обработан!')

print('Все файлы обработаны.')
