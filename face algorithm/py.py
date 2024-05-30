from deepface import DeepFace
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# 处理图像的函数
def process_image(file_i):
    try:
        img_i = np.array(Image.open(file_i))
        img = Image.fromarray(img_i)
        img = img.convert('RGB')
        img.save(os.path.join('Labeleddb', str(file_i.name)))
    except Exception as e:
        print(f"错误: {e}")


# 查找脸部的函数
def find_face(i):
    res_list = []
    if i.endswith(".jpg"):
        try:
            res = DeepFace.find(os.path.join('Labeleddb', i), db_path='Labeleddb/', threshold=1, silent=False,
                                enforce_detection=False)
            for row in range(len(res[0])):
                filebase = os.path.basename(res[0]['identity'][row])
                res_list.append((i[:-4], filebase[0:-4], res[0]['distance'][row]))  # 使用适当方法获取文件名
        except Exception as e:
            print(f"错误: {e}")
    return res_list


# 并行执行任务的函数
def run_parallel_tasks(task_function, task_args_list):
    num_cores = cpu_count()
    num_processes = max(2, num_cores//4)  # 保证至少有两个进程
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(task_function, task_args_list), total=len(task_args_list)))
    return results


# 将部分结果保存到CSV文件的函数
def save_results_to_csv(results, csv_file_path):
    if len(results) > 0:  # 检查是否有要保存的结果
        df = pd.DataFrame(results, columns=["File1", "File2", "Distance"])
        if not os.path.isfile(csv_file_path):  # 如果文件不存在，则创建并添加头部
            df.to_csv(csv_file_path, index=False)
        else:
            df.to_csv(csv_file_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    if not os.path.isdir('Labeleddb'):
        os.makedirs('Labeleddb')

    csv_file_path = os.path.join("file-file-dis", "2-Labeled-face-dataset.csv")
    if not os.path.exists(os.path.dirname(csv_file_path)):  # 确保存储CSV的目录存在
        os.makedirs(os.path.dirname(csv_file_path))

    database = r'Labeled_Faces_in_the_Wild_dataset'
    image_files = [file for subdir in os.listdir(database) for file in Path(os.path.join(database, subdir)).glob("*.*")]

    # 并行处理图像并保存
    process_results = run_parallel_tasks(process_image, image_files)

    # 获取处理后的图像列表
    img_list = os.listdir('Labeleddb')
    total_images = len(img_list)
    save_interval_percentage = 0.10
    save_interval = int(total_images * save_interval_percentage)  # 设定保存的间隔
    csv_file_path = os.path.join("file-file-dis", "2-Labeled-face-dataset.csv")

    # 分批处理图片并行查找脸部
    for batch_start in tqdm(range(0, total_images, save_interval), desc="Batch Processing Faces"):
        # 获取当前批次要处理的图像
        current_batch = img_list[batch_start:batch_start + save_interval]
        # 并行查找脸部
        batch_results = run_parallel_tasks(find_face, current_batch)

        # 汇总结果，并保存到CSV文件中
        accumulated_results = [item for sublist in batch_results for item in sublist]
        save_results_to_csv(accumulated_results, csv_file_path)
        accumulated_results.clear()  # 清空累积的结果列表以开始新的批次
