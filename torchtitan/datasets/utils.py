from typing import Any, List


def add_padding(
    token_seq: List[Any],
    padding_item: Any,
    target_len: int,
) -> List[Any]:

    padding_len = target_len - len(token_seq)
    assert (
        padding_len >= 0
    ), "target_len must be greater than or equal to the length of token_seq"
    output = token_seq + [padding_item] * padding_len
    return output
