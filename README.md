# NLPs 개요
- NLPs: LLM agent(번역, 챗봇, 분석 등)조합 on-premise 형태로 개발 및 GPU 서버 설치
- 관련자료: NLPs 개발계획.pptx

<details>
<summary>system architecture</summary>
<div markdown="1">

<img width="824" alt="20241206_092613" src="https://github.com/user-attachments/assets/f797a105-2f4e-4d4b-bcfb-c737414a7dba">

</div>
</details>

<details>
<summary>security</summary>
<div markdown="1">

<img width="905" alt="20241206_092629" src="https://github.com/user-attachments/assets/56a98e51-da5b-4b85-9a9b-dafc8fde6155">

</div>
</details>





# NLPs prototype 개요
- NLPs prototype: 금융 분야 특화 번역 agent 개발 테스트
- NLPs prototype 제작 목적
```sh
  1. NLPs의 구현 가능성 확인
  2. agent fine-tuning 및 serving 시 한계 파악 및 성능 향상 방안 연구
```
- 관련자료: NLPs prototype 개발제안서.pptx


# NLPs prototype LLM 선정
- 선정 LLM: Llama-3.1-70B-Instruct
- LLM 선정 근거: Llama, Qwen, Mistral 기본성능 정성 평가 결과, GPU memory 사용량, inference 속도 비교
- 관련자료: NLPs prototype LLM 및 GPU 선정.pptx, NLPs prototype LLM 기본성능 테스트.xlsx

<details>
<summary>LLM 기본 성능 비교 예시</summary>
<div markdown="1">

<img width="736" alt="20241206_114802" src="https://github.com/user-attachments/assets/44855ed1-2e39-40ed-b09a-ffd4e14dff0c">
<img width="706" alt="20241206_114808" src="https://github.com/user-attachments/assets/f740a114-79e5-4e80-937c-81dbb4d53ee4">

</div>
</details>



# NLPs prototype Hardware 선정
- 선정 Hardware

| 품명 | 수량 |
|------|--------|
|Intel® Xeon® Silver 4516Y+ (24Core, 2.2GHz, 45M, 185W)|2|
|NVIDIA Ada L40S 48GB GDDR6 PCIe|4|
|Samsung SSD PM893 1.92TB, 2.5in|2|
|32G DDR5 RDIMM|16|

- Hardware 선정 근거: fine-tuning, inference시 RAM, GPU memory 사용량 등 비교 (개발 GPU서버가 없어 fine-tuning 성능도 파악함)
- 관련자료: NLPs prototype LLM 및 GPU 선정.pptx, NLPs prototype GPU 성능 테스트.pptx


# NLPs prototype fine-tuning



