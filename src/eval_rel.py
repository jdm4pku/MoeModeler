import os
import json
import ast

def compute_f1(predict_path):
    result = {
        "interface":[0,0,0],
        "requirements reference":[0,0,0],
        "requirements constraints":[0,0,0]
    }
    with open(predict_path,'r',encoding='utf-8') as file:
        predict_data = json.load(file)
    for predict_item in predict_data:
        predict = ast.literal_eval(predict_item["predict"])
        ground = ast.literal_eval(predict_item["ground"])
        print("================")
        print(predict)
        for i,key in enumerate(result.keys()):
            flat_predicted_set = {tuple(sorted(pair)) for pair in predict[key]}
            flat_ground_set = {tuple(sorted(pair)) for pair in ground[key]}
            TP = len(flat_predicted_set & flat_ground_set)
            FP = len(flat_predicted_set - flat_ground_set)
            FN = len(flat_ground_set - flat_predicted_set)
            result[key][0] += TP
            result[key][1] += FP
            result[key][2] += FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for key,value in result.items():
        TP = value[0]
        FP = value[1]
        FN = value[2]
        total_tp += TP
        total_fp += FP
        total_fn += FN
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall !=0 else 0
        value.append(precision)
        value.append(recall)
        value.append(f1)
    print(result)
    total_p = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0
    total_r = total_tp / (total_tp + total_fn) if total_tp + total_fn !=0 else 0
    total_f1 = 2 * (total_p * total_r) / (total_p + total_r) if total_p + total_r!=0 else 0
    print(f"total_p:{total_p}, total_r:{total_r}, total_f1:{total_f1}")

def main():
    predict_path = "/home/jindongming/project/modeling/MoeModeler/predict/MixRE/qwen2/fold_0/relation.json"
    compute_f1(predict_path)

if __name__=="__main__":
    main()