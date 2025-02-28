import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# 读取灰度图像 S055L0421
image_path = './image/0405_6475/S048L0284.jpg'
image = Image.open(image_path).convert('L')  # 将图像转换为灰度图像（如果已经是灰度图，这一步不会有影响）

# 将图像转换为numpy数组
image_array = np.array(image)

# 将图像数据展平成一维数组
pixel_intensities = image_array.flatten()



image_path_1 = './image/lamp/S2001L05.jpg'
image_1 = Image.open(image_path_1).convert('L')  # 将图像转换为灰度图像（如果已经是灰度图，这一步不会有影响）
# 将图像转换为numpy数组
image_array_1 = np.array(image_1)
# 将图像数据展平成一维数组
pixel_intensities_1 = image_array_1.flatten()


# stacked_images = np.vstack(pixel_intensities)
image_path_2 = './image/lamp/S2062L02.jpg'
image_2 = Image.open(image_path_2).convert('L')  # 将图像转换为灰度图像（如果已经是灰度图，这一步不会有影响）
image_array_2 = np.array(image_2)
pixel_intensities_2 = image_array_2.flatten()


# 使用Seaborn绘制KDEPlot
plt.figure(figsize=(10, 6))
# sns.kdeplot(pixel_intensities, label='0405', fill=True, )
sns.kdeplot(pixel_intensities_1, label='ubiris',  fill=True)
sns.kdeplot(pixel_intensities_2, label='0405', fill=True)
plt.legend([ "source: ND-IRIS-0405", "target: 0405"])
# 设置标题和标签
# plt.title('Pixel Intensity Distribution (KDEPlot)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')

# 显示图像
plt.show()