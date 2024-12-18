import json
from time import time

class converter_1to1:
    def __init__(self, json_path):
        start = time()
        with open(json_path, "r")as f:
            self.check_data = json.load(f)
        end = round(time() - start, 4)
        print(f"{json_path} for convert_1to1 loading completed in {end} seconds")

    def search_1to1_candidate(self, user_input, depth):
        search_result_dict = {}
        for check in self.check_data[depth]:
            if check in user_input:
                search_result_dict[check] = self.check_data[depth][check]
    
        return search_result_dict

    def run(self, user_input_lst, iter_flag):
        final_lst = []
        candidate_dict = {}

        for user_input in user_input_lst:
            candidate_dict.update(self.search_1to1_candidate(user_input, "first"))
            candidate_dict.update(self.search_1to1_candidate(user_input, "second"))

            if len(candidate_dict) == 0: # 후보가 없는 경우
                final_lst.append(user_input) # 추후에 재검사 알고리즘 적용 가능

            else: # 후보가 있는 경우
                if iter_flag:
                    # 후보 Dict의 Key를 길이순으로 정렬하여 모두 교체
                    sorted_items = sorted(candidate_dict.items(), key=lambda item: len(item[0]), reverse=True)
                    for kor_replace_candi, eng_replace_candi in sorted_items:
                        user_input = user_input.replace(kor_replace_candi, eng_replace_candi)
                    final_lst.append(user_input)

                else:
                    # 후보 Dict의 Key가 가장 긴 하나만을 교체
                    longest_candidate = max(candidate_dict.keys(), key=len)
                    user_input = user_input.replace(longest_candidate, candidate_dict[longest_candidate])   
                    final_lst.append(user_input)

        return final_lst