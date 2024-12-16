import torch

# Инициализация параметров
w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001
num_iterations = 500
# Функция для вычисления значения
def f(w):
    return torch.prod(torch.log(torch.log(w + 7)))
# Градиентный спуск
for _ in range(num_iterations):
    # Вычисляем значение функции
    function = f(w)
    # Вычисляем градиент
    function.backward()
    # Обновляем веса
    with torch.no_grad():
        w -= alpha * w.grad
    # Обнуляем градиенты для следующей итерации
    w.grad.zero_()
# Выводим результат
#print(w)