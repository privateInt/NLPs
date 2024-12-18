from typing import List, Callable

async def search(
    input_text: str,
    input_dict: dict,
    key_name: str,
    value_name: str,
) -> List[dict]:
    result_lst = []
    for key in input_dict:
        tmp_dict = {}
        if key in input_text:
            tmp_dict[key_name] = key
            tmp_dict[value_name] = input_dict[key]
            result_lst.append(tmp_dict)
    return result_lst