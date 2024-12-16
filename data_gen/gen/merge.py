import json
import glob
import torch
import os

def merge_json_files(directory, pattern, output_file):
    """
    合并指定目录下所有符合给定pattern的JSON文件, 并将结果保存到output_file中。

    :param directory: 文件所在目录
    :param pattern: 文件匹配模式, 如'split_*_prob.json'
    :param output_file: 合并后的JSON文件保存路径
    """
    # 拼接目录路径和文件模式
    search_pattern = os.path.join(directory, pattern)
    json_files = sorted(glob.glob(search_pattern))
    merged_list = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_list.extend(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, indent=4)
    print(f'JSON files merged into {output_file}')

def merge_pt_files(directory, pattern, output_file):
    """
    合并指定目录下所有符合给定pattern的PT文件，并将结果保存到output_file中。

    :param directory: 文件所在目录
    :param pattern: 文件匹配模式, 如'split_*.pt'
    :param output_file: 合并后的PT文件保存路径
    """
    # 拼接目录路径和文件模式
    search_pattern = os.path.join(directory, pattern)
    pt_files = sorted(glob.glob(search_pattern))
    tensors_list = []

    for pt_file in pt_files:
        tensor = torch.load(pt_file, weights_only=True)
        tensors_list.append(tensor)

    merged_tensor = torch.cat(tensors_list, dim=0)
    torch.save(merged_tensor, output_file)
    print(f'PT files merged into {output_file}')

# 使用示例
merge_json_files('data/generated/ultrafeedback/gemma-2b-it/split_n5', 'split_*_prob.json', 'data/generated/ultrafeedback/gemma-2b-it/merged_prob.gemma.json')
merge_json_files('data/generated/ultrafeedback/llama-3b-it/split_n5', 'split_*_prob.json', 'data/generated/ultrafeedback/llama-3b-it/merged_prob.llama.json')
# merge_json_files('/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/SimPO/data/ultrafeedback/2b/s4', 'split_*_prob_sl.json', 'merged_prob_sl.json')
# merge_json_files('/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/SimPO/data/ultrafeedback/2b/s4', 'split_*_prob_sl_2b_vllm.json', 'merged_prob_sl_2b_vllm.json')
# merge_pt_files('/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/SimPO/data/ultrafeedback/2b/s4', 'split_*.pt', 'merged_tensor.pt')
