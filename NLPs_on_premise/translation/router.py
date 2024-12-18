from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_utils import get_app_state_attribute

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schema import TranslateRequest, TranslateResponse, SearchRequest, SearchResponse
from model import translator
from service import translate_text, search_keyword

router = APIRouter()

# 의존성 주입 함수
def get_translator(request: Request) -> translator:
    return get_app_state_attribute(
        request=request,
        attr_name="translator",
        error_message="translator instance is not initialized."
    )

def get_eng_key_dict(request: Request) -> dict:
    return get_app_state_attribute(
        request=request,
        attr_name="eng_key_dict",
        error_message="eng_key_dict for search keyword has not been loaded."
    )

def get_kor_key_dict(request: Request) -> dict:
    return get_app_state_attribute(
        request=request,
        attr_name="kor_key_dict",
        error_message="kor_key_dict for search keyword has not been loaded."
    )

# router 정의
@router.post("/translate", response_model=TranslateResponse)
async def translate(
    request: TranslateRequest,
    translator: translator = Depends(get_translator)
):
    """
    한국어 금융 데이터를 영어로 번역하는 router입니다.

    Example:
    
        Request Body:
        {
            "user_input": "감자전(원)"
        }

        Response Body:
        {
            "result": true,
            "translation": "Before capital reduction (KRW)"
        }
    """
    try:
        translate_result = await translate_text(
            user_input = request.user_input, 
            translator = translator
        )
        
        return TranslateResponse(
            result = True, 
            translation = str(translate_result),
        )
        
    except Exception as e:
        return JSONResponse(
            status_code = 500, 
            content = {
                "result": False, 
                "error": str(e)
            }
        )

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    eng_key_dict: dict = Depends(get_eng_key_dict),
    kor_key_dict: dict = Depends(get_kor_key_dict),
):
    """
    입력된 한국어 금융 데이터(text)에서 keyword를 추출합니다.
    미리 가지고 있는 금융 사전에서 해당 단어가 있는지 검사하여 List에 추가합니다.

    Example:
        Request Body:
        {
            "user_input": "4. Details of debt guarantee"
        }

        {
            "result": true,
            "keyword_lst": [
                {
                  "eng": "Details",
                  "kor": "내용"
                },
                {
                    "eng": "Details of debt guarantee",
                    "kor": "채무보증내역"
                },
                {
                    "eng": "Details of debt",
                    "kor": "채무내용"
                },
                {
                    "eng": "debt",
                    "kor": "부채, 차입, 채무"
                },
                {
                    "eng": "guarantee",
                    "kor": "담보"
                }
            ]
        }
    
    """
    try:
        keyword_lst = await search_keyword(
            user_input = str(request.user_input),
            eng_key_dict = eng_key_dict,
            kor_key_dict = kor_key_dict
        )
        
        return SearchResponse(
            result = True, 
            keyword_lst = keyword_lst
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "result": False, 
                "error": str(e)
            }
        )