import torch
import torch.nn as nn

class SineNet(nn.Module):
    def __init__(self, n_hidden_neurons):
        super().__init__()
        self.fc1 = nn.Linear(1, n_hidden_neurons)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden_neurons, n_hidden_neurons)  # Добавлен слой
        self.act2 = nn.Tanh()  # Добавлен слой
        self.fc3 = nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

# Создаем сеть с 20 нейронами в скрытом слое
sine_net = SineNet(int(input()))

# Создаем тестовый входной тензор
x = torch.tensor([1.])

# Прямой проход и вывод информации о сети
output = sine_net(x)
print(sine_net)