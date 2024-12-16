import torch

# Определяем целевую функцию
def target_function(x):
    return 2**x * torch.sin(2**-x)

# Определяем архитектуру нейронной сети
class RegressionNet(torch.nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, 64)  # Уменьшенный размер входного слоя
        self.fc2 = torch.nn.Linear(64, 32)  # Уменьшенный размер скрытого слоя
        self.fc3 = torch.nn.Linear(32, 1)   # Выходной слой
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

net = RegressionNet()

# ------ Подготовка данных --------
x_train = torch.linspace(-10, 5, 100).unsqueeze(1)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.0
y_train += noise

x_validation = torch.linspace(-10, 5, 100).unsqueeze(1)
y_validation = target_function(x_validation)
# ------ Подготовка данных --------

# Определяем оптимизатор и скорость обучения
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Определяем функцию потерь (MAE)
def loss(pred, target):
    return torch.mean(torch.abs(pred - target))

# Обучение модели
n_epochs = 500  # Уменьшено количество эпох
for epoch in range(n_epochs):
    optimizer.zero_grad()  # Обнуляем градиенты

    y_pred = net(x_train)  # Прямой проход
    loss_value = loss(y_pred, y_train)  # Вычисляем значение потерь
    loss_value.backward()  # Обратный проход
    optimizer.step()  # Шаг оптимизации

# Проверка метрики
def metric(pred, target):
    return (pred - target).abs().mean()

# Печатаем значение MAE на валидации
mae = metric(net.forward(x_validation), y_validation)
#print(mae.item())