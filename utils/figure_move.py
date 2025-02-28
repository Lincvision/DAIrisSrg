import os
import shutil

# 设置源文件夹和目标文件夹的路径
num_train = 1198
num_train_u = 1200
# num_valid  = (num_train)//7
num_valid  = 268
source_folder = 'I:/对比试验/deeplab/dataset/1-16/Lamp_16212/test'  # test_GT
source_folder_gt = 'I:/对比试验/deeplab/dataset/1-16/Lamp_16212/test_GT'  # test_GT

train_folder = os.path.join('I:/对比试验/deeplab/dataset', str(num_train), 'train')
train_folder_gt = os.path.join('I:/对比试验/deeplab/dataset', str(num_train), 'train_GT')
valid_folder = os.path.join('I:/对比试验/deeplab/dataset', str(num_train), 'valid')
valid_folder_gt = os.path.join('I:/对比试验/deeplab/dataset', str(num_train), 'valid_GT')

train_u_folder = os.path.join('I:/对比试验/deeplab/dataset', str(num_train), 'train_u')
train_u_folder_gt = os.path.join('I:/对比试验/deeplab/dataset', str(num_train), 'train_u_GT')
# 获取源文件夹中所有文件的列表，并对其进行排序
files_image = sorted(os.listdir(source_folder))
files_gt = sorted(os.listdir(source_folder_gt ))

# 确保目标文件夹存在
os.makedirs(train_folder, exist_ok=True)
os.makedirs(train_folder_gt, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(valid_folder_gt, exist_ok=True)
os.makedirs(train_u_folder, exist_ok=True)
os.makedirs(train_u_folder_gt, exist_ok=True)

# 复制前1200张图片到train文件夹
for i in range(0, num_train):
    file_name = files_image[i]
    file_name_gt = files_gt[i]

    source_path = os.path.join(source_folder, file_name)
    train_path = os.path.join(train_folder, file_name)
    shutil.copy(source_path, train_path)

    source_path_gt = os.path.join(source_folder_gt, file_name_gt)
    train_path_gt = os.path.join(train_folder_gt, file_name_gt)
    shutil.copy(source_path_gt, train_path_gt)

# 复制前1200张图片到train文件夹
for i in range(num_train, num_train+num_valid):
    file_name = files_image[i]
    file_name_gt = files_gt[i]

    source_path = os.path.join(source_folder, file_name)
    train_path = os.path.join(valid_folder, file_name)
    shutil.copy(source_path, train_path)

    source_path_gt = os.path.join(source_folder_gt, file_name_gt)
    train_path_gt = os.path.join(valid_folder_gt, file_name_gt)
    shutil.copy(source_path_gt, train_path_gt)


for i in range(num_train+num_valid, num_train+num_valid+num_train_u):
    file_name = files_image[i]
    file_name_gt = files_gt[i]

    source_path = os.path.join(source_folder, file_name)
    train_path = os.path.join(train_u_folder, file_name)
    shutil.copy(source_path, train_path)

    source_path_gt = os.path.join(source_folder_gt, file_name_gt)
    train_path_gt = os.path.join(train_u_folder_gt, file_name_gt)
    shutil.copy(source_path_gt, train_path_gt)


print('图片复制完成。')