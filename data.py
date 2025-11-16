import os
import shutil
import glob

def reorganize_dataset(base_path, dir_name):
    """
    주어진 디렉토리(train 또는 val)의 데이터셋 구조를 재구성합니다.

    Args:
        base_path (str): 데이터셋의 루트 경로.
        dir_name (str): 처리할 디렉토리 이름 ('train' 또는 'val').
    """
    print(f"--- '{dir_name}' 디렉토리 처리 중 ---")
    
    # 경로 정의
    current_dir_path = os.path.join(base_path, dir_name)
    new_jpg_path = os.path.join(current_dir_path, 'jpg')
    new_json_path = os.path.join(current_dir_path, 'json')

    # 새로운 디렉토리 생성
    os.makedirs(new_jpg_path, exist_ok=True)
    print(f"디렉토리 생성 완료: {new_jpg_path}")
    os.makedirs(new_json_path, exist_ok=True)
    print(f"디렉토리 생성 완료: {new_json_path}")

    # 원본 디렉토리 패턴 정의
    source_jpg_dirs = [os.path.join(current_dir_path, 'report_jpg'), os.path.join(current_dir_path, 'press_jpg')]
    source_json_dirs = [os.path.join(current_dir_path, 'report_json'), os.path.join(current_dir_path, 'press_json')]

    # JPG 파일 이동
    for src_dir in source_jpg_dirs:
        if not os.path.isdir(src_dir):
            print(f"경로를 찾을 수 없어 건너뜁니다: {src_dir}")
            continue
        
        files_to_move = glob.glob(os.path.join(src_dir, '*.jpg'))
        if not files_to_move:
            print(f"{src_dir} 에서 JPG 파일을 찾을 수 없습니다.")
        else:
            print(f"{src_dir}의 JPG 파일 {len(files_to_move)}개를 {new_jpg_path}로 이동합니다...")
            for file_path in files_to_move:
                file_name = os.path.basename(file_path)
                destination_path = os.path.join(new_jpg_path, file_name)
                shutil.move(file_path, destination_path)
        
        # 비어있는 원본 디렉토리 삭제
        try:
            os.rmdir(src_dir)
            print(f"빈 디렉토리 삭제 완료: {src_dir}")
        except OSError as e:
            print(f"디렉토리를 삭제할 수 없습니다 {src_dir}: {e}")

    # JSON 파일 이동
    for src_dir in source_json_dirs:
        if not os.path.isdir(src_dir):
            print(f"경로를 찾을 수 없어 건너뜁니다: {src_dir}")
            continue

        files_to_move = glob.glob(os.path.join(src_dir, '*.json'))
        if not files_to_move:
            print(f"{src_dir} 에서 JSON 파일을 찾을 수 없습니다.")
        else:
            print(f"{src_dir}의 JSON 파일 {len(files_to_move)}개를 {new_json_path}로 이동합니다...")
            for file_path in files_to_move:
                file_name = os.path.basename(file_path)
                destination_path = os.path.join(new_json_path, file_name)
                shutil.move(file_path, destination_path)
        
        # 비어있는 원본 디렉토리 삭제
        try:
            os.rmdir(src_dir)
            print(f"빈 디렉토리 삭제 완료: {src_dir}")
        except OSError as e:
            print(f"디렉토리를 삭제할 수 없습니다 {src_dir}: {e}")
            
    print(f"--- '{dir_name}' 디렉토리 처리 완료 ---\n")


if __name__ == '__main__':
    # 데이터셋의 기본 경로를 설정합니다.
    # 중요: 실제 데이터셋 위치에 맞게 이 경로를 수정해주세요.
    BASE_DATASET_PATH = '/home/elicer/train_valid'

    # 'train'과 'val' 디렉토리 구조를 재구성합니다.
    reorganize_dataset(BASE_DATASET_PATH, 'train')
    reorganize_dataset(BASE_DATASET_PATH, 'valid')

    print("모든 디렉토리 구조 변경이 성공적으로 완료되었습니다.")