# NLPs 개요
NLPs란?
```sh
LLM agent(번역, 챗봇, 분석 등)조합을 on-premise로 개발 후 GPU서버에 설치하여 납품
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
  
<img width="762" alt="20241206_162330" src="https://github.com/user-attachments/assets/056eab4e-822c-4feb-85c3-d00f19a140a2">
<img width="781" alt="20241206_162316" src="https://github.com/user-attachments/assets/41ce07ee-a8d4-47c2-b0c0-2280083e81ac">
<img width="784" alt="20241206_162322" src="https://github.com/user-attachments/assets/ad9183d3-e2ea-4690-b4ea-24eab3f4084b">

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
<img width="352" alt="20241211_091535" src="https://github.com/user-attachments/assets/663067b8-40da-40b9-bd86-ac19b1c9e82e">

</div>
</details>



# NLPs prototype fine-tuning 1차 실험
hyper parameter

| 항목 | 수치 및 내용 |
|------|--------|
|model|meta-llama/Llama-3.1-70B-Instruct|
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
|model|meta-llama/Llama-3.1-70B-Instruct, allganize/Llama-3-Alpha-Ko-8B-Instruct|
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

한국어 8B fine-tuning model inference 예시 (input: 한국어, output: 영어)

<img width="415" alt="20241208_195727" src="https://github.com/user-attachments/assets/d8666492-e845-4c25-bc66-5c127499a6d6">
<img width="699" alt="20241208_195105" src="https://github.com/user-attachments/assets/5f137c64-d1d7-4cae-8dcf-a779711d9936">

실험 결과 및 개선 사항

<img width="281" alt="20241207_170647" src="https://github.com/user-attachments/assets/5fcc170a-2d36-46bb-a33f-c5dfe603b6f6">
<img width="363" alt="20241207_170641" src="https://github.com/user-attachments/assets/a5061cae-ad3b-4125-8b52-feb24d60a1ca">
<img width="361" alt="20241207_170635" src="https://github.com/user-attachments/assets/fb94a1c4-25b1-434b-971e-07c6fd5e51e1">


```sh
# 결과
1. batch_size 증가로 loss 진동폭 안정화 됨
2. epoch는 2 ~ 3정도가 충분했으며, 그 이상은 overfitting 현상을 보임
3. 한국어로 fine-tuning된 8B 모델과 70B 모델의 차이가 크지 않았음
   => 8B 모델로 실험 진행후 실험 결과는 70B에 적용하는 것이 시간 효율을 극대화할 수 있다 판단
4. input이 짧은 경우 품질이 좋았으나, input이 길어질수록 품질 저하 현상 발생

# 개선을 위한 가설
1. quantization 수치(4bit)가 너무 낮았을 가능성이 있음
2. 문장이 길어질수록 사용하는 단어가 많기때문에 학습만으로 부족할 수 있음

# 개선 계획
1. quantization 8bit 실험 후 추가 실험 필요 여부 판단
2. RAG(bge-m3)를 사용해 학습 데이터와 inference prompt에 적절한 예시 추가
```

<details>
<summary>관련자료 미리보기 (NLPs prototype fine-tuning 결과.xlsx, NLPs prototype fine-tuning 결과 발표.pptx)</summary>
<div markdown="1">

<img width="858" alt="20241208_203642" src="https://github.com/user-attachments/assets/e36ea864-1128-4c08-a5e2-05656f152168">
<img width="792" alt="20241208_203705" src="https://github.com/user-attachments/assets/7d1b9a7f-a563-40e1-af17-92e344da72dc">
<img width="760" alt="20241208_203711" src="https://github.com/user-attachments/assets/cba40302-848e-4ecb-8dff-c443b8fa40e4">

</div>
</details>



# NLPs prototype API 버전
NLPs prototype API 버전이란?
```sh
회사 일정으로 인해
openAI 등 API를 사용해 기존에 만들었던 번역시스템에
RAG를 적용해 금융 분야 특화 번역 agent 개발 테스트
```

RAG 구현 과정
```sh
1. embedding model (openAI text-embedding-3-small) 선정
2. cosine similarity 적용 방식 변경 (iteration -> matrix)
   => iteration으로 구현시 multi-processing 적용하여 실행 시간 이득을 볼 수 있을거라 생각했으나,
      데이터 복제에 시간이 더 소요돼 matrix연산으로 변경
3. np.vstack으로 vectorDB 구축 후 fastAPI의 lifespan을 통해 서버 실행시 vectorDB 로드를 한번만 수행
4. 반드시 특정 단어로 번역돼야 하는 사전 데이터를 우선 적용 후 RAG 적용
```

RAG 개선 사항
```sh
1. faiss, milvus 등 vectorDB 적용하여 실행시간 비교 필요 
   (현재는 np형태이며 계산도 직접짠 계산식으로 연산)
2. vectorDB의 중복을 제거하여 실행 속도 향상 필요
3. classification 등 방법을 통해 vectorDB를 분리하여 실행 속도 향상 필요 
   (classification 기준은 해당 분야 전문가와 협의 필요)
```

<details>
<summary>관련자료 미리보기 (NLPs prototype API embedding 모델 선정.docx, NLPs prototype API RAG 효과 분석 .docx, NLPs prototype API RAG 결과 예시.zip)</summary>
<div markdown="1">

```sh
# NLPs prototype API RAG 결과 예시.zip 파일 구조 예시
# 문서 난이도 별로 상, 중, 하로 구분되며 각 예시 내부에
# 한국어 원본 (파일명 끝에 아무것도 붙지 않음),
# 영어 정답본 (파일명 끝에 -한영C 붙음),
# RAG 적용 inference 예시 (파일명 끝에 _4o_rag_trans 붙음),
# RAG 미적용 inference 예시 (파일명 끝에 _trans 붙음)
# 총 4가지 파일이 있다.
folder
├── 상
├── 중
└── 하
    ├── CJ제일제당 결산실적공시 예고(안내공시) (2024.04.23)
    └── 기아 기업설명회(IR) 개최(안내공시) (2024.07.02)
          ├── 기아 기업설명회(IR) 개최(안내공시) (2024.07.02).xlsx (한국어 원본)
          ├── 기아 기업설명회(IR) 개최(안내공시) (2024.07.02)-한영C.xlsx (영어 정답본)
          ├── 기아 기업설명회(IR) 개최(안내공시) (2024.07.02)_4o_rag_trans.xlsx (RAG 적용 inference 예시)
          └── 기아 기업설명회(IR) 개최(안내공시) (2024.07.02)_trans.xlsx (RAG 미적용 inference 예시)
```

<img width="459" alt="20241209_185027" src="https://github.com/user-attachments/assets/ebd8259d-d609-4abb-8806-c271a47b16e5">
<img width="792" alt="20241209_184959" src="https://github.com/user-attachments/assets/d54037ec-b0d5-40cc-a5fa-45bed89456a8">
<img width="452" alt="20241209_185036" src="https://github.com/user-attachments/assets/67505f19-0cd1-4220-be60-bdf05b1470d2">

</div>
</details>

# 향후 계획
1. on-premise agent 개발 process 정립
2. pre-processing 고도화 (ex. document parsing)
3. backend 고도화 (ex. kubernetes)
