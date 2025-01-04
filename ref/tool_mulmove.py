import os
import shutil
import re
import time

#从2347移出来
tic = time.time()

src_folder = "/mnt/truenas_datasets/Datasets_EM/mul2347_mie_train"
dest_folder = "/mnt/truenas_datasets/Datasets_EM/RCS/pb943_mie_train"
pattern = r"RCSmap_theta(\d+)phi(\d+)f(\d+\.\d+).pt"
myplane = 'b943'
# pattern = r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d+\.\d+).pt"
# cnt = 1880 #抽出1/10做验证集
os.makedirs(dest_folder, exist_ok=True)

files = os.listdir(src_folder)
for file_name in files:
    match = re.match(pattern, file_name)
    if match:
        plane = match.group(1)
        theta = int(match.group(2))
        phi = int(match.group(3))
        f = float(match.group(4))
        # print(f)
        if plane == myplane:
            src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)
            shutil.move(src_file_path, dest_file_path)
    #         cnt -= 1
    # if cnt == 0:
    #     break

total_size = 0
for root, dirs, files in os.walk(dest_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        total_size += os.path.getsize(file_path)

total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB

print(f"目标文件夹中的文件数量：{len(os.listdir(dest_folder))}, {total_size_gb:.2f} GB")
print(f'耗时{(time.time()-tic):.4f}s')
trainfolder = dest_folder

#从train移成val
tic = time.time()

src_folder = "/mnt/truenas_datasets/Datasets_EM/RCS/pb943_mie_train"
dest_folder = "/mnt/truenas_datasets/Datasets_EM/RCS/pb943_mie_val"
pattern = r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d+\.\d+).pt"
cnt = 1880 #抽出1/10做验证集
os.makedirs(dest_folder, exist_ok=True)

files = os.listdir(src_folder)
for file_name in files:
    match = re.match(pattern, file_name)
    if match:
        plane = match.group(1)
        theta = int(match.group(2))
        phi = int(match.group(3))
        f = float(match.group(4))
        # print(f)
        if f <= 1.0:
            src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)
            shutil.move(src_file_path, dest_file_path)
            cnt -= 1
    if cnt == 0:
        break

total_size = 0
for root, dirs, files in os.walk(dest_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        total_size += os.path.getsize(file_path)

total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB

print(f"目标文件夹中的文件数量：{len(os.listdir(dest_folder))}, {total_size_gb:.2f} GB")
print(f'耗时{(time.time()-tic):.4f}s')

#从train拷贝成pretrain
tic = time.time()

src_folder = "/mnt/truenas_datasets/Datasets_EM/RCS/pb943_mie_train"
dest_folder = "/mnt/truenas_datasets/Datasets_EM/RCS/pb943_mie_pretrain"
pattern = r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d+\.\d+).pt"
cnt = 940 #抽出1/10做验证集
os.makedirs(dest_folder, exist_ok=True)

files = os.listdir(src_folder)
for file_name in files:
    match = re.match(pattern, file_name)
    if match:
        plane = match.group(1)
        theta = int(match.group(2))
        phi = int(match.group(3))
        f = float(match.group(4))
        # print(f)
        if f <= 1.0:
            src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)
            shutil.copy(src_file_path, dest_file_path)
            cnt -= 1
    if cnt == 0:
        break

total_size = 0
for root, dirs, files in os.walk(dest_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        total_size += os.path.getsize(file_path)

total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB

print(f"目标文件夹中的文件数量：{len(os.listdir(dest_folder))}, {total_size_gb:.2f} GB")
print(f'耗时{(time.time()-tic):.4f}s')


print(f"train目标文件夹中的文件数量：{len(os.listdir(trainfolder))}")
total_size = 0
for root, dirs, files in os.walk(trainfolder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        total_size += os.path.getsize(file_path)

total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB
print(f"目标文件夹大小：{total_size_gb:.2f} GB")