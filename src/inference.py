import os
import json
from openai import OpenAI
from logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

def write_json(file_path,data):
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)

def get_completion(prompt):
    client = OpenAI(
        api_key = "0",
        base_url = "http://localhost:8000/v1"
    )
    response = client.chat.completions.create(
        messages=[{"role":"user","content":prompt}],
        model = "test"
    )
    return response.choices[0].message.content

def get_predict(input_dir,output_dir):
    with open(input_dir,'r',encoding='utf-8') as file:
        test_data = json.load(file)
    final_result = []
    for item in tqdm(test_data):
        instruction = item["instruction"]
        sentence = item["input"]
        ground = item["output"]
        prompt = instruction + sentence
        response = get_completion(prompt)
        predict_dict = {
            "input":sentence,
            "ground":ground,
            "predict":response
        }
        logger.info(response)
        final_result.append(predict_dict)

    write_json(output_dir,final_result)

def main():
    input_dir = "/home/jindongming/project/modeling/MoeModeler/instruct-dataset/relation/10-fold/fold_0/test.json"
    output_dir = "/home/jindongming/project/modeling/MoeModeler/predict/MixER/fold_0/relation.json"
    get_predict(input_dir,output_dir)

if __name__=="__main__":
    main()
