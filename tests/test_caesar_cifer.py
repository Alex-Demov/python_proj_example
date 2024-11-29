# python
from typing import LiteralString

# project
from src.tools.caesar_cifer import caesar_cifer


def test_case_one() -> None:
    input_string_one: LiteralString = "python"
    stride_one: int = 4
    assert caesar_cifer(input_string_one, stride_one) == "tbxlsr"


def test_case_two() -> None:
    input_string_two: LiteralString = "programming"
    stride_two: int = -3
    assert caesar_cifer(input_string_two, stride_two) == "moldoyjjfkd"
