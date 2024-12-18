from prompt import prompt_dict
from utils import search
from model import translator
from typing import List

async def translate_text(
    user_input: str, 
    translator: translator
) -> str:
    system_message = prompt_dict["system_message"]
    
    try:
        # 여기에 translator를 사용하여 실제 번역 작업을 수행합니다
        translate_result = await translator.inference(
            system_message=system_message,
            user_input=user_input
        )

        return translate_result
    
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}")

async def search_keyword(
    user_input: str, 
    eng_key_dict: dict, 
    kor_key_dict: dict
) -> List[str]:
    try:
        eng_key_lst = await search(
            user_input, 
            eng_key_dict, 
            "eng", 
            "kor", 
        )
        
        kor_key_lst = await search(
            user_input,
            kor_key_dict, 
            "kor", 
            "eng", 
        )

        return eng_key_lst+kor_key_lst
        
    except Exception as e:
        raise RuntimeError(f"Searching keyword failed: {str(e)}")