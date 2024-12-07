# NLPs 개요
NLPs란?
```sh
LLM agent(번역, 챗봇, 분석 등)조합 on-premise 개발 및 GPU 서버 설치
```

<details>
<summary>관련자료 미리보기 (NLPs 개발계획.pptx)</summary>
<div markdown="1">

<img width="824" alt="20241206_092613" src="https://github.com/user-attachments/assets/f797a105-2f4e-4d4b-bcfb-c737414a7dba">
<img width="905" alt="20241206_092629" src="https://github.com/user-attachments/assets/56a98e51-da5b-4b85-9a9b-dafc8fde6155">

</div>
</details>



# NLPs prototype 개요
NLPs prototype이란?
```sh
금융 분야 특화 번역 agent 개발 테스트
```
NLPs prototype 제작 목적
```sh
1. NLPs의 구현 가능성 확인
2. agent fine-tuning 및 serving 시 한계 파악 및 성능 향상 방안 연구
```

<details>
<summary>관련자료 미리보기 (NLPs prototype 개발제안서.pptx)</summary>
<div markdown="1">
  
<img width="784" alt="20241206_162322" src="https://github.com/user-attachments/assets/ad9183d3-e2ea-4690-b4ea-24eab3f4084b">
<img width="781" alt="20241206_162316" src="https://github.com/user-attachments/assets/41ce07ee-a8d4-47c2-b0c0-2280083e81ac">
<img width="762" alt="20241206_162330" src="https://github.com/user-attachments/assets/056eab4e-822c-4feb-85c3-d00f19a140a2">

</div>
</details>



# NLPs prototype LLM 선정
선정 LLM
```sh
Llama-3.1-70B-Instruct
```
LLM 선정 근거
```sh
Llama, Qwen, Mistral 기본성능 정성 평가 결과, GPU memory 사용량, inference 속도 비교
```

<details>
<summary>관련자료 미리보기 ( NLPs prototype LLM 및 GPU 선정.pptx, NLPs prototype LLM 기본성능 테스트.xlsx)</summary>
<div markdown="1">

<img width="736" alt="20241206_114802" src="https://github.com/user-attachments/assets/44855ed1-2e39-40ed-b09a-ffd4e14dff0c">
<img width="706" alt="20241206_114808" src="https://github.com/user-attachments/assets/f740a114-79e5-4e80-937c-81dbb4d53ee4">

</div>
</details>



# NLPs prototype Hardware 선정
선정 Hardware

| 품명 | 수량 |
|------|--------|
|Intel® Xeon® Silver 4516Y+ (24Core, 2.2GHz, 45M, 185W)|2|
|NVIDIA Ada L40S 48GB GDDR6 PCIe|4|
|Samsung SSD PM893 1.92TB, 2.5in|2|
|32G DDR5 RDIMM|16|

Hardware 선정시 고려사항
```sh
1. 가격 및 구입 소요 시간
2. fine-tuning, inference시 RAM, GPU memory 사용량
3. LLM 용량
```

<details>
<summary>관련자료 미리보기 (NLPs prototype LLM 및 GPU 선정.pptx, NLPs prototype GPU 성능 테스트.pptx)</summary>
<div markdown="1">

<img width="792" alt="20241206_162914" src="https://github.com/user-attachments/assets/59925681-af60-4364-a574-77448b0b00a6">
<img width="777" alt="20241206_163140" src="https://github.com/user-attachments/assets/6a127c6f-59d3-4dd0-8270-9d007fc0e93d">
<img width="969" alt="20241206_163205" src="https://github.com/user-attachments/assets/76540b27-d294-4d0c-81ba-21d57cf00314">
<img width="922" alt="20241206_163242" src="https://github.com/user-attachments/assets/d05796fc-084e-4dd6-84b1-b440accc33a8">

</div>
</details>



# NLPs prototype fine-tuning 1차 실험
hyper parameter

| 항목 | 수치 및 내용 |
|------|--------|
|LoRA_r|16|
|QLoRA|4bit|
|dtype|bf16|
|epoch|3|
|per_device_train_batch_size|2|
|gradient_accumulation_steps|4|
|learning_rate|2e-4|
|warmup_steps|1000|
|GPU memory usage|약 43GB|

실험 결과 및 개선 사항

<img width="499" alt="20241207_114905" src="https://github.com/user-attachments/assets/188d0688-ec47-4a36-99a1-fe3ff9759b00">
<img width="211" alt="20241207_165834" src="https://github.com/user-attachments/assets/94b77e3e-d3ee-40a9-842d-8ce1173514f0">
<img width="483" alt="스크린샷 2024-09-11 오전 9 10 43" src="https://github.com/user-attachments/assets/d6a5fd62-39bb-4476-8b5c-119a0f671923">

```sh
# 결과
Loss 진동폭이 너무 커지는 현상 발생 => 학습이 잘 안될 수 있으므로 Loss 진동폭 감소 방안 필요

# 개선을 위한 가설
1. 배치 크기(batch_size & gradient_accumulate_step)가 너무 작음
2. 데이터의 일관성 부족
* 학습률도 loss 진동폭에 영향을 미칠 수 있으나 그래프가 안정적이므로 적절하다 판단함.

# 개선 계획
1. 배치 크기 등 hyper parameter는 금방 검증할 수 있으므로 1순위로 검증한다.
2. hyper parameter tuning이 효과가 없는 경우 데이터의 중복(같은 한국어를 다른 영어로 번역한 데이터가 있는 경우 등)을 검사한다.
```



# NLPs prototype fine-tuning 2차 실험
hyper parameter

| 항목 | 수치 및 내용 |
|------|--------|
|LoRA_r|16|
|QLoRA|4bit|
|dtype|bf16|
|epoch|10|
|per_device_train_batch_size|32|
|gradient_accumulation_steps|4|
|learning_rate|2e-4|
|warmup_steps|1000|
|GPU memory usage|약 69GB|

테스트 목록
```sh
1. per_device_train_batch_size: 2 -> 32 (현재 GPU에서 작동 가능 여부, Loss 진동폭 감소 여부 확인)
2. epoch: 3 -> 10 (이후 epoch별 check point의 성능을 점검하여 최적의 epoch 선정)
3. 한국어로 fine-tuning된 8B 모델(allganize/Llama-3-Alpha-Ko-8B-Instruct) 추가 테스트 
```

실험 결과 및 개선 사항


