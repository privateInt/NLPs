# NLPs 개요

- NLPs: 고객 요청에 따른 LLM agent(번역, 챗봇, 분석 등)조합을 on-premise 형태로 개발후, GPU 서버에 설치하여 납품
- 관련자료: NLPs 개발계획.pptx


<img width="824" alt="20241206_092613" src="https://github.com/user-attachments/assets/f797a105-2f4e-4d4b-bcfb-c737414a7dba">

<img width="905" alt="20241206_092629" src="https://github.com/user-attachments/assets/56a98e51-da5b-4b85-9a9b-dafc8fde6155">


# NLPs prototype 개요

- NLPs prototype: 금융 분야 특화 번역 agent 개발 테스트
- NLPs prototype 제작 목적
```sh
  1. NLPs의 구현 가능성 확인
  2. agent fine-tuning 및 serving 시 한계 파악 및 성능 향상 방안 연구
```




# 파일 구조
```sh
project
├── requirements.txt
├── run_ocr.py
├── ssh_ocr.py
├── ssh_ocr_API.py
└── ssh_ocr_utils.py
```

# 파일 역할
| 파일 | 설명 |
|------|--------|
|requirements|환경 설치 파일|
|run_ocr.py|streamlit 실행파일(ssh_ocr.py)을 subprocess로 실행|
|ssh_ocr.py|streamlit 실행파일|
|ssh_ocr_API.py|OCR API 관련 정보가 저장돼 있다. API틀을 base class로 작성했으며 이후 추가되는 API class는 base class를 상속하여 사용|
|ssh_ocr_utils|압축파일 해제, OCR결과 1줄 정리 등 다양한 utils들이 정의돼 있다.|

# 명령어

<table border="1">
  <tr>
    <th>내용</th>
    <th>명령어</th>
  </tr>
  <tr>
    <td>환경 설치</td>
    <td>pip install -r requirements.txt</td>
  </tr>
  <tr>
    <td>streamlit page 실행</td>
    <td>python run_ocr.py</td>
  </tr>
</table>

# 데모 페이지 예시

![스크린샷 2024-09-24 오전 10 35 33](https://github.com/user-attachments/assets/b04c5e55-ace7-4d71-b69c-48aec2e9ceb3)
