import logging
from typing import Callable, Any
from fastapi import HTTPException, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_resource(
    task: Callable[[], Any],  # 초기화 작업 (함수)
    success_message: str,     # 성공 시 메시지
    error_message: str        # 실패 시 메시지
) -> Any:
    """초기화 작업과 예외 처리를 공통적으로 처리하는 함수"""
    try:
        result = task()  # 초기화 작업 실행
        logger.info(success_message)
        return result
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        raise RuntimeError(error_message) from e

def get_app_state_attribute(request: Request, attr_name: str, error_message: str) -> Any:
    """app state 호출하는 함수"""
    app = request.app
    if not hasattr(app.state, attr_name):
        raise HTTPException(status_code=500, detail=error_message)
    return getattr(app.state, attr_name)