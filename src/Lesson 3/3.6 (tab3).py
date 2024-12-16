import random

def find_seed(target):
    for seed in range(100):  # Перебираем значения seed от 0 до 99
        random.seed(seed)
        if random.randint(0, 10) == target:
            print(target) #print(f"Подходящее значение seed: {seed} (возвращает {target})")
            break

find_seed(5)  # Ищем seed для получения 5