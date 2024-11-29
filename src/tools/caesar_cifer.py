# python
from string import ascii_lowercase
from typing import LiteralString


def caesar_cifer(input_string: str, stride: int) -> str:
    """Функция, реализующая шифр Цезаря

    Возвращает:
        * input_string (`str`): входная строка
        * stride (`int`): сдвиг

    Возвращает:
        * `str`: зашифрованная строка
    """
    updated_ascii_lowercase: LiteralString = " " + ascii_lowercase

    result = ""
    for sym in input_string:
        index_sym: int = updated_ascii_lowercase.index(sym)
        new_index_sym: int = (index_sym + stride) % len(updated_ascii_lowercase)
        result += updated_ascii_lowercase[new_index_sym]

    return result
