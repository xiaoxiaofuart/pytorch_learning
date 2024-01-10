from PIL import Image
import  numpy as np
import  cv2
import matplotlib.pyplot as plt

# 计算机中彩色图片是有CHW组成，C表示通道，H表示高度，W表示宽度
imag_ndarray_res =cv2.imread("1550546614997.jpg")

imag_ndarray = np.array(imag_ndarray_res)

# 获取第一个通道数据
imag_r = imag_ndarray[:, :, 0]

# 获取第二个通道数据
imag_g = imag_ndarray[:, :, 1]

# 获取第三个通道数据
imag_b = imag_ndarray[:, :, 2]

cv2.imshow(imag_r)
# 显示原始图像和各个通道
# plt.subplot(221), plt.imshow(imag_ndarray), plt.title('Original Image')
# plt.subplot(222), plt.imshow(imag_b, cmap='Blues'), plt.title('Blue Channel')
# plt.subplot(223), plt.imshow(imag_g, cmap='Greens'), plt.title('Green Channel')
# plt.subplot(224), plt.imshow(imag_r, cmap='Reds'), plt.title('Red Channel')
# plt.show()