import json
import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from main_utils import initialize_resource
from translation.router import router as translation_router
from translation.model import translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # translation 초기 데이터 로드
    app.state.translator = initialize_resource(
        task=lambda: translator(
            model_path="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        ),
        success_message="Translator loading complete!",
        error_message="Translator loading failed"
    )

    ## keyword search eng JSON 파일 로드
    app.state.eng_key_dict = initialize_resource(
        task=lambda: json.load(open("translation/eng_key_dict.json", "r", encoding="utf-8")),
        success_message="English dictionary (search_dict_eng) loading complete!",
        error_message="English dictionary loading failed"
    )

    ## keyword search kor 파일 생성
    app.state.kor_key_dict = initialize_resource(
        task=lambda: {kor: eng for eng, kor in app.state.eng_key_dict.items()},
        success_message="Korean dictionary (search_dict_kor) loading complete!",
        error_message="Korean dictionary creation failed"
    )

    yield

    # 자원 정리
    for attr in ["translator", "search_dict_eng", "search_dict_kor"]:
        if hasattr(app.state, attr):
            delattr(app.state, attr)
            logger.info(f"App state '{attr}' has been cleaned up.")

app = FastAPI(
    root_path="/dev",
    lifespan=lifespan,
)
            

# Configure CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)

# Include the routes
app.include_router(translation_router)

# Server execution
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=12211, reload=True)