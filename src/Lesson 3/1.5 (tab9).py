import torch

# Определяем функцию
def f(w):
    return torch.prod(torch.log(torch.log(w + 7)))
# Создаем тензор с заданными значениями и преобразуем в тип с плавающей точкой
w = torch.tensor([[5, 10], [1, 2]], dtype=torch.float32, requires_grad=True)
# Вычисляем значение функции
function = f(w)
# Вычисляем градиент
function.backward()
# Выводим градиент
#print(w.grad)