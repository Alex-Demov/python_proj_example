# project
from src.tools.caesar_cifer import caesar_cifer


def main() -> None:
    stride: int = int(input("Введите сдвиг шифрования: "))
    input_string: str = input("Введите строку для шифрования: ").strip()

    result: str = caesar_cifer(input_string, stride)
    print(f'Результат: "{result}"')


if __name__ == "__main__":
    main()
