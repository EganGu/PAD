import json
import math
import os
import argparse

def split_json_file(input_file, k, output_dir):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算每个文件的大小
    n = len(data)
    size_per_file = math.ceil(n / k)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 分割数据并写入新的JSON文件
    for i in range(k):
        start_index = i * size_per_file
        end_index = min(start_index + size_per_file, n)
        split_data = data[start_index:end_index]

        output_file = os.path.join(output_dir, f'split_{i+1}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=4)

        print(f'Created {output_file} with {len(split_data)} elements.')

def merge_json_files(output_file, input_files):
    merged_data = []
    
    # 读取并合并所有JSON文件
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
    
    # 写入合并后的数据到一个新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f'Merged files into {output_file} with {len(merged_data)} elements.')

def main():
    parser = argparse.ArgumentParser(description="Split and merge JSON files.")
    
    parser.add_argument('-k', type=int, required=True, help='Number of files to split into')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split files')

    args = parser.parse_args()
    
    # 拆分文件
    split_json_file(args.input_file, args.k, args.output_dir)
    
    # # 合并文件示例（可选）
    # input_files = [os.path.join(args.output_dir, f'split_{i+1}.json') for i in range(args.k)]
    # output_file = 'merged_output.json'
    # merge_json_files(output_file, input_files)

if __name__ == '__main__':
    main()
