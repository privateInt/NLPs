import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# API key 설정
API_KEY = 'api_key'

# 데이터 로드
with open("370k_merged_data_240923.json", "r") as f:
    data = json.load(f)

# OpenAI API 설정
client = OpenAI(api_key=API_KEY)

# get_embedding 함수 정의
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# 각 데이터를 처리하는 함수 정의
def process_data(item, model_lst):
    new_item = item.copy()
    for model in model_lst:
        new_item["kor_embed_" + model.split("-")[-1]] = get_embedding(new_item["kor"], model)
    return new_item

# 모델 리스트 설정
model_lst = ["text-embedding-3-small", "text-embedding-3-large"]

# 멀티스레딩을 사용하여 데이터 병렬 처리
new_lst = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_data, item, model_lst) for item in data]
    for future in tqdm(as_completed(futures), total=len(futures)):
        new_lst.append(future.result())

# 새 파일 저장
with open("370k_with_embed_data_241105.json", "w") as f:
    json.dump(new_lst, f)
