from pydantic import BaseModel
from typing import List

# set BaseModel
class TranslateRequest(BaseModel):
    user_input: str

class TranslateResponse(BaseModel):
    result: bool
    translation: str

class SearchRequest(BaseModel):
    user_input: str

class SearchResponse(BaseModel):
    result: bool
    keyword_lst: List[dict]