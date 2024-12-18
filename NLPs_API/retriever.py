import json
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from time import time
from typing import List, Dict

class retriever:
    def __init__(
            self, 
            json_path: str, 
            embedded_df_column_name: str,
            model_name: str,
            api_key: str,
        ):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        
        self.embedded_df_column_name = embedded_df_column_name
        
        start = time()
        self.df = pd.read_json(json_path)
        self.embedded_df_array = np.vstack(self.df[self.embedded_df_column_name].values)
        end = round(time() - start, 4)
        print(f"{json_path} for retriever loading completed in {end} seconds")

    # get_embedding 함수 정의 (텍스트 리스트에 대해 임베딩 반환)
    def get_embeddings(self, texts: List) -> np.array:
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = [self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding for text in texts]
        return np.array(embeddings)

    # search 함수 정의 (N개의 입력에 대해 top_k 유사도 반환)
    def search(self, kor_input_data_list: List, top_k: int) -> dict:
        process_start_time = time()
        
        # N개의 입력을 임베딩으로 변환
        embedded_kor_input_data_array = self.get_embeddings(kor_input_data_list)

        # 코사인 유사도를 벡터화된 방식으로 계산
        dot_products = np.dot(self.embedded_df_array, embedded_kor_input_data_array.T)  # (M x N)
        norm_products = np.outer(
            np.linalg.norm(self.embedded_df_array, axis=1),  # M차원
            np.linalg.norm(embedded_kor_input_data_array, axis=1)  # N차원
        )  # (M x N)
        
        cosine_similarities = dot_products / norm_products  # (M x N)

        # 각 입력에 대해 top_k 유사도 추출
        results = []
        for idx, input_cosine_similarities in enumerate(cosine_similarities.T):
            self.df["cosine_similarity"] = input_cosine_similarities
            search_output_df = self.df.sort_values("cosine_similarity", ascending=False).head(top_k)
            
            relevant_examples_dict = {
                row["cosine_similarity"]: {"kor": row["kor"], "eng": row["eng"]}
                for _, row in search_output_df.iterrows()
            }
            
            results.append({
                "kor_input_data": kor_input_data_list[idx],
                "embedded_kor_input_data": embedded_kor_input_data_array[idx].tolist(),
                "relevant_examples_dict": dict(sorted(relevant_examples_dict.items(), reverse=True))
            })

        process_end_time = round(time() - process_start_time, 4)
        return {
            "results": results,
            "processing_time": process_end_time
        }