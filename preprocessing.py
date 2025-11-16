import pandas as pd
import json
import os
from glob import glob
from tqdm import tqdm

def preprocess_to_csv(base_path, dir_name, output_csv_name):
    """
    Processes JSON files from a specified directory (train or val) and saves them as a CSV file.
    (train/val 용 원본 함수)
    """
    # [수정] train/val은 'json' 하위 폴더를 사용합니다.
    json_dir = os.path.join(base_path, dir_name, 'json')
    json_files = glob(os.path.join(json_dir, '*.json'))
    
    if not json_files:
        print(f"Warning: No JSON files found in '{json_dir}'. Please ensure the folder structure is correct.")
        return

    print(f"Preprocessing {len(json_files)} JSON files from '{dir_name}' directory...")
    
    all_rows = []

    for file_path in tqdm(json_files, desc=f"Processing {dir_name} JSONs"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            source_info = data.get('source_data_info', {})
            learning_info = data.get('learning_data_info', {})
            
            source_data_name_jpg = source_info.get('source_data_name_jpg')
            visual_context = learning_info.get('visual_context')

            annotations = learning_info.get('annotation', [])
            for anno in annotations:
                class_id = anno.get('class_id', '')
                
                if class_id and class_id.startswith('V'):
                    row = {
                        'source_data_name_jpg': source_data_name_jpg,
                        'visual_context': visual_context,
                        'class_name' : anno.get('class_name'),
                        'visual_instruction': anno.get('visual_instruction'),
                        'visual_answer': anno.get('visual_answer'),
                        'bounding_box': anno.get('bounding_box') # train/val은 bounding_box 포함
                    }
                    all_rows.append(row)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing file {file_path}: {e}")

    if not all_rows:
        print(f"No valid data to process in '{dir_name}' directory.")
        return
        
    df = pd.DataFrame(all_rows)
    
    columns_order = [
        'source_data_name_jpg',
        'visual_context',
        'class_name',
        'visual_instruction',
        'visual_answer',
        'bounding_box' # train/val은 bounding_box 포함
    ]
    
    # 누락된 열이 있어도 오류가 나지 않도록 처리
    df = df.reindex(columns=columns_order)

    output_path = os.path.join(base_path, output_csv_name)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nPreprocessing complete! {len(df)} rows saved to '{output_path}'.")
    print("First 5 rows of the generated CSV:")
    print(df.head())
    print("-" * 80)


def preprocess_test_to_csv(base_path, dir_name, json_sub_dir, output_csv_name):
    """
    Processes JSON files from the test directory ('test/query') and saves them as a CSV file.
    (test용 신규 함수)
    """
    # [수정] test는 'query' 하위 폴더를 사용합니다.
    json_dir = os.path.join(base_path, dir_name, json_sub_dir)
    json_files = glob(os.path.join(json_dir, '*.json'))
    
    if not json_files:
        print(f"Warning: No JSON files found in '{json_dir}'. Please ensure the folder structure is correct.")
        return

    print(f"Preprocessing {len(json_files)} JSON files from '{dir_name}/{json_sub_dir}' directory...")
    
    all_rows = []

    for file_path in tqdm(json_files, desc=f"Processing {dir_name} JSONs"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            source_info = data.get('source_data_info', {})
            learning_info = data.get('learning_data_info', {})
            
            source_data_name_jpg = source_info.get('source_data_name_jpg')
            visual_context = learning_info.get('visual_context')

            annotations = learning_info.get('annotation', [])
            for anno in annotations:
                class_id = anno.get('class_id', '')
                
                if class_id and class_id.startswith('V'):
                    row = {
                        'source_data_name_jpg': source_data_name_jpg,
                        'visual_context': visual_context,
                        'class_name' : anno.get('class_name'),
                        'visual_instruction': anno.get('visual_instruction'),
                        'visual_answer': anno.get('visual_answer'),
                        'instance_id' : anno.get('instance_id')
                        # [수정] test는 'bounding_box'를 포함하지 않습니다.
                    }
                    all_rows.append(row)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing file {file_path}: {e}")

    if not all_rows:
        print(f"No valid data to process in '{dir_name}' directory.")
        return
        
    df = pd.DataFrame(all_rows)
    
    # [수정] 'bounding_box'가 빠진 컬럼 순서
    columns_order = [
        'source_data_name_jpg',
        'visual_context',
        'class_name',
        'visual_instruction',
        'visual_answer',
        'instance_id'
    ]
    df = df.reindex(columns=columns_order)

    output_path = os.path.join(base_path, output_csv_name)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nTest preprocessing complete! {len(df)} rows saved to '{output_path}'.")
    print("First 5 rows of the generated CSV:")
    print(df.head())
    print("-" * 80)


if __name__ == '__main__':
    # --- Configuration ---
    # 데이터셋이 있는 기본 경로
    BASE_DATASET_PATH = '/home/elicer/train_valid' # 현재 폴더 기준 (경로가 다르면 수정하세요, 예: '/home/elicer/train_valid')

    # # --- Execution ---
    # # 1. 'train' 디렉토리 처리 (train/json -> train.csv)
    # preprocess_to_csv(BASE_DATASET_PATH, 'train', 'train.csv')

    # # 2. 'val' 디렉토리 처리 (val/json -> val.csv)
    # # 'val' 폴더가 'valid'일 경우 'val'을 'valid'로 수정하세요
    # preprocess_to_csv(BASE_DATASET_PATH, 'valid', 'val.csv') 

    # 3. 'test' 디렉토리 처리 (test/query -> test.csv)
    preprocess_test_to_csv(BASE_DATASET_PATH, 'test', 'query', 'test.csv')

    print("All preprocessing tasks have been successfully completed.")