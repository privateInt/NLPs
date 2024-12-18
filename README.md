<details>
<summary>NLPs 프로젝트 개요</summary>
<div markdown="1">

# NLPs 개요
- NLPs란?
```sh
고객요청에 따른 LLM agent(번역, 챗봇, 분석 등)조합을 on-premise 또는 API로 개발하여
GPU서버에 설치후 GPU서버 자체를 판매하는 프로젝트
```

<br>

<details>
<summary>관련자료 미리보기 (NLPs 개발계획.pptx)</summary>
<div markdown="1">

<img width="824" alt="20241206_092613" src="https://github.com/user-attachments/assets/f797a105-2f4e-4d4b-bcfb-c737414a7dba">
<img width="905" alt="20241206_092629" src="https://github.com/user-attachments/assets/56a98e51-da5b-4b85-9a9b-dafc8fde6155">

</div>
</details>

<br><br>


# NLPs prototype 개요
- NLPs prototype이란?
```sh
NLPs 개발 과정에서의 어려움, 노하우 등을 미리 파악하기 위해
금융 분야 특화 번역 agent 시제품 개발 테스트
```

<br>

- NLPs prototype 제작 목적
```sh
1. NLPs의 구현 가능성 확인
2. agent fine-tuning 및 serving 시 한계 파악 및 성능 향상 방안 연구
```

<br>

<details>
<summary>관련자료 미리보기 (NLPs prototype 개발제안서.pptx)</summary>
<div markdown="1">
  
<img width="762" alt="20241206_162330" src="https://github.com/user-attachments/assets/056eab4e-822c-4feb-85c3-d00f19a140a2">
<img width="781" alt="20241206_162316" src="https://github.com/user-attachments/assets/41ce07ee-a8d4-47c2-b0c0-2280083e81ac">
<img width="784" alt="20241206_162322" src="https://github.com/user-attachments/assets/ad9183d3-e2ea-4690-b4ea-24eab3f4084b">

</div>
</details>

<br><br>


# NLPs prototype on-premise LLM 선정
- 선정 LLM
```sh
Llama-3.1-70B-Instruct
```

<br>

- 선정 근거
```sh
Llama, Qwen, Mistral 기본성능 정성 평가 결과, GPU memory 사용량, inference 속도 비교
```

<br>

<details>
<summary>관련자료 미리보기 ( NLPs prototype LLM 및 GPU 선정.pptx, NLPs prototype LLM 기본성능 테스트.xlsx)</summary>
<div markdown="1">

<img width="736" alt="20241206_114802" src="https://github.com/user-attachments/assets/44855ed1-2e39-40ed-b09a-ffd4e14dff0c">
<img width="706" alt="20241206_114808" src="https://github.com/user-attachments/assets/f740a114-79e5-4e80-937c-81dbb4d53ee4">

</div>
</details>

<br><br>


# NLPs prototype Hardware 선정
- 선정 Hardware

| 품명 | 수량 |
|------|--------|
|Intel® Xeon® Silver 4516Y+ (24Core, 2.2GHz, 45M, 185W)|2|
|NVIDIA Ada L40S 48GB GDDR6 PCIe|4|
|Samsung SSD PM893 1.92TB, 2.5in|2|
|32G DDR5 RDIMM|16|

<br>

- Hardware 선정시 고려사항
```sh
1. 가격 및 구입 소요 시간
2. fine-tuning, inference시 RAM, GPU memory 사용량
3. LLM 용량
```

<br>

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

<br><br>

</div>
</details>













# 프로젝트 목적

- 금융 분야 특화 번역기 fine-tuning (한 -> 영)
- 이 project에서는 금융에 관련된 주제를 다루며, 데이터 형식을 맞춘다면 다른 domain에도 적용 가능

# 기능

- LLM을 fine-tuning 할 수 있다. unsloth를 사용했으며 LoRA, QLoRA 등 PEFT가 가능하다.(default: meta-llama/Llama-3.1-8B-Instruct)
- 금융분야 관련 user input(한국어) 입력시 영어 번역문을 출력할 수 있다.
- 한국어, 영어로 구성된 사전을 통해 user input(한국어, 영어) 입력시 keyword를 추출할 수 있다.
- fine-tining 결과 및 FastAPI 서버로 띄울 수 있다.

# 파일 구조
```sh
NLPs_on_premise
├── requirements.txt
├── main.py
├── main_utils.py
└── fine-tuning 
    ├── train.py
    └── data
         └── dataset.pkl
└── translation
    ├── eng_key_dict.json
    ├── model.py
    ├── prompt.py
    ├── router.py
    ├── schema.py
    ├── service.py
    └── utils.py
```

# 폴더 및 파일 역할
| 폴더 및 파일 | 설명 |
|------|--------|
|requirements.txt|project 작동시 필요한 library를 모아놓은 txt파일|
|main.py|LLM의 fine-tuning 결과를 이용해 inference server(FastAPI)를 작동|
|main.py|lifespan 등 FastAPI 관련 utils 정의|
|fine-tuning|fine-tuning 및 데이터를 저장하는 폴더|
|fine-tuning/data|LLM fine-tuning 데이터를 위치시키는 폴더, pkl파일을 읽고 처리하는 방식이며, 데이터셋은 List[dict] 형태.|
|fine-tuning/train.py|LLM fine-tuning 코드, unsloth 사용|
|translation|한->영 금융 번역 LLM agent 관련 코드 저장|
|translation/eng_key_dict.json|keyword searching에 사용되는 한국어-영어 사전 데이터|
|translation/model.py|LLM load 및 inference 코드를 class로 정의|
|translation/prompt.py|한->영 금융 번역에 사용되는 prompt 정의 (추후 model과 prompt에 따라 다양한 번역 가능)|
|translation/router.py|router 정의|
|translation/schema.py|router의 입출력 정의|
|translation/service.py|router를 통해 호출받으면 model.py, utils.py 등의 기능을 조합한 service를 정의|
|translation/utils.py|keyword searching 알고리즘 등 utils 정의|


# 환경
- GPU: A100(80GiB)GPU
- python 3.10
- CUDA Version 12.6
- Nvidia Driver Version 560.35.03
  
<img width="472" alt="20241218_220821" src="https://github.com/user-attachments/assets/caef58da-7d6f-42bf-b12a-0aabeb15f8ae" />



# 학습 실험

## fine-tuning 1차 실험
- hyper parameter

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

GPU memory usage 약 43GB

<br>

- 실험 결과 및 개선 사항

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

<br><br>



## fine-tuning 2차 실험
- hyper parameter

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

GPU memory usage 약 69GB

<br>

- 테스트 목록
```sh
1. per_device_train_batch_size: 2 -> 32 (현재 GPU에서 작동 가능 여부, Loss 진동폭 감소 여부 확인)
2. epoch: 3 -> 10 (이후 epoch별 check point의 성능을 점검하여 최적의 epoch 선정)
3. 한국어로 fine-tuning된 8B 모델(allganize/Llama-3-Alpha-Ko-8B-Instruct) 추가 테스트 
```

<br>

<details>
<summary>한국어 8B fine-tuning model inference 예시 (input: 한국어, output: 영어)</summary>
<div markdown="1">

<img width="415" alt="20241208_195727" src="https://github.com/user-attachments/assets/d8666492-e845-4c25-bc66-5c127499a6d6">
<img width="699" alt="20241208_195105" src="https://github.com/user-attachments/assets/5f137c64-d1d7-4cae-8dcf-a779711d9936">

</div>
</details>

<br>


- 실험 결과 및 개선 사항

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
2. 문장이 길어질수록 사용하는 단어가 많기때문에 SFT 학습만으로 부족할 수 있음

# 개선 계획
1. quantization 8bit 실험 후 추가 실험 필요 여부 판단
2. RAG(bge-m3)를 사용해 학습 데이터와 inference prompt에 적절한 예시 추가
3. chatGPT 등을 사용하여 DPO 학습 데이터 구축후 DPO 학습 실험
```

<br>

<details>
<summary>관련자료 미리보기 (NLPs prototype fine-tuning 결과.xlsx, NLPs prototype fine-tuning 결과 발표.pptx)</summary>
<div markdown="1">

<img width="858" alt="20241208_203642" src="https://github.com/user-attachments/assets/e36ea864-1128-4c08-a5e2-05656f152168">
<img width="792" alt="20241208_203705" src="https://github.com/user-attachments/assets/7d1b9a7f-a563-40e1-af17-92e344da72dc">
<img width="760" alt="20241208_203711" src="https://github.com/user-attachments/assets/cba40302-848e-4ecb-8dff-c443b8fa40e4">

</div>
</details>

<br><br>

## DPR
- hyper parameter
  
| hyper parameter | value |
|------|--------|
|epochs|20|
|batch_size|16|
|learning_rate|3e-4|
|passage_chunk_size|100|

- GPU memory usage(fine-tuning): 약 25GiB
- GPU memory usage(inference): 약 4GiB
- DPR 논문에 의하면 batch_size를 크게할 수록 negative sample을 증가시켜 fine-tuning 성능을 향상시킬 수 있다.
- DPR 논문에 의하면 passage를 chunk_size로 분할하여 검색한다. 이 project에서는 정보 손실을 고려하여 분할된 chunk를 원복하여 LLM에 제공한다.

## LLM
- hyper parameter
  
| hyper parameter | value |
|------|--------|
|lora_r|32|
|lora_alpha|lora_r * 2|
|epochs|20|
|batch_size|128|
|micro_batch_size|8|
|learning_rate|3e-5|

- GPU memory usage(fine-tuning): 약 60GiB
- GPU memory usage(inference): 약 56GiB
- lora값에 따라 trainable parameter가 결정됐다. lora논문에 의하면 trainable parameter가 0.2%여도 성능에 큰 차이가 없다는 사실을 확인하기 위해, lora_r의 값은 32, 3072로 조정하여 각 trainable parameter를 0.2%, 16.3%로 변경하여 fine-tuning 성능을 비교했지만 큰 차이를 발견할 수 없었다.
- 데이터 수와 품질의 관계를 파악하기 위해 데이터 수가 많지만 품질이 떨어지는 데이터셋, 데이터 수가 적지만 품질이 좋은 데이터셋을 각각 fine-tuning하여 성능을 비교했다. 비교 결과 데이터 수가 적더라도 품질이 뛰어나야 LLM의 성능에 긍정적이라는 사실을 확인했다.

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
    <td rowspan="3">DPR 학습</td>
    <td>cd [YOUR WORKSPACE]</td>
  </tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>cd DPR</td>
  </tr>
  <tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>python DPR_trainer.py</td>
  </tr>
  <tr>
    <td rowspan="3">LLM 학습</td>
    <td>cd [YOUR WORKSPACE]</td>
  </tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>cd LLM</td>
  </tr>
  <tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>python LLM_trainer.py</td>
  </tr>
  <tr>
    <td>run server & demo page</td>
    <td>sh run.sh</td>
  </tr>
  <tr>
    <td rowspan="2">docker image build</td>
    <td>cd [YOUR WORKSPACE]</td>
  </tr>
  <tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>docker build -t kullm:ver2 .</td>
  </tr>
  <tr>
    <td>docker container start</td>
    <td>docker-compose up -d</td>
  </tr>
  
</table>

# 데모 페이지 예시

![Cap 2024-05-25 14-55-04-185](https://github.com/privateInt/RAG-chatbot/assets/95892797/1357398d-5ac2-4492-8f51-adad63d75b20)


# 한계 및 극복방안

- 데이터 부분: 이 project에서는 사람이 직접 데이터를 제작했다. 추후 일반화하기 위해서는 데이터 추출 자동화가 선행돼야 한다.
- Retrieval 성능: top1의 acc는 30%에 그쳤다. Retrieval 성능을 올리기 위해 BM25같은 sparse retriever와 ensemble retriever 구축이 필요하다. 또한 보조 모델을 활용하는 등 topK가 커져도 gold passage를 추출하는 기능이 필요하다.
- LLM 성능: Retrieval이 잘못된 정보를 전달할 경우 LLM이 필터링할 수 있어야 한다.

# 앞으로의 방향

![Cap 2024-02-13 16-59-36-937](https://github.com/privateInt/RAG-chatbot/assets/95892797/d6161384-2ccf-4b8b-b644-cf5fe1da77de)

![Cap 2024-02-13 17-00-56-248](https://github.com/privateInt/RAG-chatbot/assets/95892797/1778fc2f-26f0-42b1-a878-fb4e0a2a5fe9)

## 데이터 pipeline
- 문서에는 자연어와 표, 이미지 등이 섞여있음
- 표의 경우 MD, HTML 등의 형태로 변경하는 등 LLM 학습 가능한 형태로 1차 가공 필요
- 1차 가공이 끝난 데이터를 출처,제시문,질문,답변 등으로 구분하는 2차 가공 필요

## advanced RAG
- Multi-Query: 질문 표현 방식에 따라 retrieval 성능이 바뀌므로 같은 의미지만 다른 표현을 여러개 생성하여 retriever 필요
- Self-Query: 조건식으로 검색 대상을 최대한 좁힌 후 retriever 시행
- Time-Weighted: meta data(문서 생성 시기, 문서 종류 등)를 활용해 검색 대상을 최대한 좁힌 후 retriever 시행
- Ensemble-Retriever: sparse retriever의 성능이 더 좋은 경우가 존재하기 때문에 구축 필요
- Long Context Reorder: 연관성이 높은 문서를 일부러 맨앞, 맨뒤에 Reorder

![Cap 2024-02-14 12-21-48-066](https://github.com/privateInt/RAG-chatbot/assets/95892797/3214e493-fb28-4af1-a190-8f7e9dac049d)

## graph DB
- advanced RAG만으로 Retriever 성능이 부족할 수 있음, 추가 retriever system 구축이 필요함
- graph DB는 노드(키워드)와 엣지(관계)로 이루어짐
- 관계 정의 가이드라인 필요
- 키워드 추출 가이드라인 필요

## chain of thought
- prompt를 단계별로 제공하여 LLM이 문제를 쉽게 이해할 수 있도록 조치 필요











# NLPs prototype API 버전
- NLPs prototype API 버전이란?
```sh
기존에 구축한 범용 번역시스템에
RAG를 적용해 금융 분야 특화 번역 agent 개발 테스트
```

<br>

- RAG 구현 과정
```sh
1. embedding model (openAI text-embedding-3-small) 선정
2. cosine similarity 적용 방식 변경 (iteration -> matrix)
   => iteration으로 구현시 multi-processing 적용하여 실행 시간 이득을 볼 수 있을거라 생각했으나,
      데이터 복제에 시간이 더 소요돼 matrix연산으로 변경
3. np.vstack으로 vectorDB 구축 후 fastAPI의 lifespan을 통해 서버 실행시 vectorDB 로드를 한번만 수행
4. 반드시 특정 단어로 번역돼야 하는 사전 데이터를 우선 적용 후 RAG 적용
```

<br>

- RAG 개선 사항
```sh
1. faiss, milvus 등 vectorDB 적용하여 실행시간 비교 필요 
   (현재는 np형태이며 계산도 직접짠 계산식으로 연산)
2. vectorDB의 중복을 제거하여 실행 속도 향상 필요
3. classification 등 방법을 통해 vectorDB를 분리하여 실행 속도 향상 필요 
   (classification 기준은 해당 분야 전문가와 협의 필요)
```

<br>

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

<br><br>
