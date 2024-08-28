import os
import json
import random

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(file_path,data):
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)
    
def read_txt(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        content = file.read()
    return content

def format_instruct_data(data):
    entity_instruct_file = "/home/jindongming/project/modeling/MoeModeler/instruct/entity.txt"
    relation_instruct_file = "/home/jindongming/project/modeling/MoeModeler/instruct/relation.txt"
    entity_instruct = read_txt(entity_instruct_file)
    relation_instruct = read_txt(relation_instruct_file)
    entity_instruct_data = []
    relation_instruct_data = []
    for item in data:
        requirements = item["text"]
        entity = item["entity"]
        item = {
            "instruction":entity_instruct,
            "input":requirements,
            "output":str(entity)
        }
        entity_instruct_data.append(item)
    for item in data:
        requirements = item["text"]
        entity = item["entity"]
        relation = item["relation"]
        item = {
            "instruction":relation_instruct,
            "input": requirements + f"\nEntity List:{str(entity)}",
            "output": str(relation)
        }
        relation_instruct_data.append(item)
    return entity_instruct_data, relation_instruct_data
    
def process_data(in_dir,output_dir):
    fold_list = ['fold_0','fold_1','fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9']
    for fold in fold_list:
        in_path = os.path.join(in_dir,fold)
        train_in_path = os.path.join(in_path,"train_data.json")
        test_in_path = os.path.join(in_path,"test_data.json")
        train_data = read_json(train_in_path)
        test_data = read_json(test_in_path)
        train_entity_ins,train_relation_ins = format_instruct_data(train_data)
        test_entity_ins,test_relation_ins = format_instruct_data(test_data)
        entity_output_dir = os.path.join(output_dir,f"entity/10-fold/{fold}")
        if not os.path.exists(entity_output_dir):
            os.makedirs(entity_output_dir)
        train_entity_output_file = os.path.join(entity_output_dir,"train.json")
        test_entity_output_file = os.path.join(entity_output_dir,"test.json")
        write_json(train_entity_output_file,train_entity_ins)
        write_json(test_entity_output_file,test_entity_ins)
        relation_output_dir = os.path.join(output_dir,f"relation/10-fold/{fold}")
        if not os.path.exists(relation_output_dir):
            os.makedirs(relation_output_dir)
        train_relation_output_file = os.path.join(relation_output_dir,"train.json")
        test_relation_output_file = os.path.join(relation_output_dir,"test.json")
        write_json(train_relation_output_file,train_relation_ins)
        write_json(test_relation_output_file,test_relation_ins)

        # 处理数据的混合
        # 先实体后关系
        train_data = train_entity_ins + train_relation_ins
        test_data = test_entity_ins + test_relation_ins
        mixture_dir = os.path.join(output_dir,f"mixture/entity-relation/10-fold/{fold}")
        if not os.path.exists(mixture_dir):
            os.makedirs(mixture_dir)
        train_path = os.path.join(mixture_dir,"train.json")
        test_path = os.path.join(mixture_dir,"test.json")
        write_json(train_path,train_data)
        write_json(test_path,test_data)
        # 先关系后实体
        train_data = train_relation_ins + train_entity_ins
        test_data = test_relation_ins + test_entity_ins
        mixture_dir = os.path.join(output_dir,f"mixture/relation-entity/10-fold/{fold}")
        if not os.path.exists(mixture_dir):
            os.makedirs(mixture_dir)
        train_path = os.path.join(mixture_dir,"train.json")
        test_path = os.path.join(mixture_dir,"test.json")
        write_json(train_path,train_data)
        write_json(test_path,test_data)
        # 打乱混合
        train_data = train_entity_ins + train_relation_ins
        test_data = test_entity_ins + test_relation_ins
        random.shuffle(train_data)
        random.shuffle(test_data)
        mixture_dir = os.path.join(output_dir,f"mixture/random/10-fold/{fold}")
        if not os.path.exists(mixture_dir):
            os.makedirs(mixture_dir)
        train_path = os.path.join(mixture_dir,"train.json")
        test_path = os.path.join(mixture_dir,"test.json")
        write_json(train_path,train_data)
        write_json(test_path,test_data)

def main():
    input_dir = "/home/jindongming/project/modeling/MoeModeler/dataset/10-fold"
    output_dir = "/home/jindongming/project/modeling/MoeModeler/instruct-dataset/"
    process_data(input_dir,output_dir)

if __name__=="__main__":
    main()