import torch

# Инициализация параметров
w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001
num_iterations = 500
# Определение функции
def f(w):
    return torch.prod(torch.log(torch.log(w + 7)))
# Создание оптимизатора
optimizer = torch.optim.SGD([w], lr=alpha)
# Градиентный спуск
for _ in range(num_iterations):
    # Вычисляем значение функции
    function = f(w)
    # Вычисляем градиенты
    optimizer.zero_grad()
    function.backward()
    # Обновляем веса
    optimizer.step()
# Выводим результат
#print(w)