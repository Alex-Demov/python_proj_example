import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import datasets, models
import pandas as pd
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Установка директорий
train_dir = r'C:\Users\Alexey\Desktop\Миленом системс\Лекция 5 (13.12.24)\Pictures_for_script\train'
test_dir = r'C:\Users\Alexey\Desktop\Миленом системс\Лекция 5 (13.12.24)\Pictures_for_script\test'
results_dir = r'C:\Users\Alexey\Desktop\Training\Training_results'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Параметры
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразование и загрузка данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Функция для обучения и тестирования модели
def train_and_evaluate(model, criterion, optimizer):
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        
        # Тестирование модели
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss = test_running_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
        
        # Сохранение модели с лучшей точностью
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), os.path.join(results_dir, f'best_model_{model.__class__.__name__}.pth'))
    
    return test_loss, test_accuracy

# Обучение и оценка моделей
results = []

# ResNet18
resnet_model = models.resnet18(pretrained=True)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
resnet_loss, resnet_accuracy = train_and_evaluate(resnet_model, criterion, optimizer)
results.append(('ResNet18', resnet_loss, resnet_accuracy))

# EfficientNet
efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = efficientnet_model.classifier[1].in_features  # Доступ к in_features правильным образом
efficientnet_model.classifier[1] = nn.Linear(num_ftrs, len(train_dataset.classes))  # Переопределение линейного слоя

optimizer = optim.Adam(efficientnet_model.parameters(), lr=0.001)
efficientnet_loss, efficientnet_accuracy = train_and_evaluate(efficientnet_model, criterion, optimizer)
results.append(('EfficientNet', efficientnet_loss, efficientnet_accuracy))

# RegNet
regnet_model = models.regnet_x_400mf(pretrained=True)
num_ftrs = regnet_model.fc.in_features
regnet_model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

optimizer = optim.Adam(regnet_model.parameters(), lr=0.001)
regnet_loss, regnet_accuracy = train_and_evaluate(regnet_model, criterion, optimizer)
results.append(('RegNet', regnet_loss, regnet_accuracy))

# Запись результатов в таблицу
results_df = pd.DataFrame(results, columns=['Model', 'Test Loss', 'Test Accuracy'])
results_df.to_csv(os.path.join(results_dir, 'model_results.csv'), index=False)

print("Обучение завершено. Результаты сохранены в 'model_results.csv'.")

# --- Новая часть кода для оценки загруженных моделей ---

# Функция для оценивания модели
def evaluate_model(model_path, model_class, num_classes):
    model = model_class(weights=None)  # Установите weights=None, чтобы не загружать предобученные веса
    num_ftrs = model.fc.in_features if hasattr(model, 'fc') else model.classifier[1].in_features
    model.fc = nn.Linear(num_ftrs, num_classes) if hasattr(model, 'fc') else nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return all_labels, all_preds

# Подсчет метрик
def calculate_metrics(y_true, y_pred):
    accuracy = Accuracy()
    precision = Precision(num_classes=len(set(y_true)), average='macro', compute_on_step=False)
    recall = Recall(num_classes=len(set(y_true)), average='macro', compute_on_step=False)
    f1_score = F1Score(num_classes=len(set(y_true)), average='macro', compute_on_step=False)

    acc = accuracy(torch.tensor(y_pred), torch.tensor(y_true))
    prec = precision(torch.tensor(y_pred), torch.tensor(y_true))
    rec = recall(torch.tensor(y_pred), torch.tensor(y_true))
    f1 = f1_score(torch.tensor(y_pred), torch.tensor(y_true))

    return acc.item(), prec.item(), rec.item(), f1.item()

# Загрузка и оценка моделей
models_to_evaluate = [
    (os.path.join(results_dir, 'best_model_ResNet18.pth'), models.resnet18),
    (os.path.join(results_dir, 'best_model_EfficientNet.pth'), models.efficientnet_b0),
    (os.path.join(results_dir, 'best_model_RegNet.pth'), models.regnet_x_400mf)
]

results_metrics = []

for model_path, model_class in models_to_evaluate:
    true_labels, pred_labels = evaluate_model(model_path, model_class, len(train_dataset.classes))
    metrics = calculate_metrics(true_labels, pred_labels)
    results_metrics.append((model_path.split('_')[2], *metrics))  # Сохраняем только название модели и метрики

# Создание DataFrame и запись результатов
metrics_df = pd.DataFrame(results_metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
metrics_df.to_csv(os.path.join(results_dir, 'model_metrics.csv'), index=False)

print("Метрики сохранены в 'model_metrics.csv'.")

