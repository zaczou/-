# 一. 软件安装
linux，ubuntu</br>
vs2015（不需要）</br>
驱动 cuda8.0(https://developer.nvidia.com/cuda-toolkit-archive), cudnn60</br>
git, anaconda</br>
python, java</br>
更改conda, pip镜像</br>
tensorflow-gpu, pytorch, keras</br>
xgboost：git, mingw or tdm-gcc</br>

pymc3, edward, pystan</br>
lingo(https://zhuanlan.zhihu.com/p/29772798), pulp</br>

word, excel, ppt</br>
bdp</br>

everything，tampermonkey</br>
sharelatex，processon</br>

# 二. python爬虫


# 三. 数据读取与保存
## 3.1 python
### 3.1.1 txt, csv等文件
``` python
f = open('', 'r')
for line in f:
```
### 3.1.2 xml文件
``` python
import xml.etree.ElementTree as ET
tree = ET.parse(fname)
root = tree.getroot()
```
### 3.1.3 图片文件
1. PIL
``` python
from PIL import Image
image = Image.open('test.jpg') ## 打开图片 宽*高
image.size #(w, h) 
image.mode #(RGB)
image = image.resize(200, 200, Image.NEAREST)
image.convert('L') ##转换灰度图
image_a = np.array(image,dtype=np.float32) # image = np.array(image)默认是uint8
image_a.shape (h,w,c)
image_a.show()
```
2. Skimage
``` python
import skimage
from skimage import io,transform
image= io.imread('test.jpg',as_grey=False) # True为灰度图, numpy.ndarray, uint8,[0-255]
image.shape #(h, w, c)
image.mode #(RGB)
transform.resize(im, output_shape=(20, 20), order=0, mode='constant', preserve_range=True).astype(np.uint8)
# order: 0 代表最近邻插值，1代表双线性插值。。。
# preserve_range: True的话表示保持原有的 取值范围，false 的话就成 0-1 了
# 返回的是 float，有需要的可以强转一下类型)
```

3. Matplotlib
``` python 
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
lena = mpimg.imread('lena.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
```

4. opencv(python版)（BGR）
``` python
import cv2
image = cv2.imread('test.jpg')
type(image) # out: numpy.ndarray
image.dtype # out: dtype('uint8')
image.shape # out: (h,w,c) 和skimage类似
cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) 
关键字参数为dst,fx,fy,interpolation
dst为缩放后的图像
dsize为(w,h),但是image是(h,w,c)
fx,fy为图像x,y方向的缩放比例，
interplolation为缩放时的插值方式，有三种插值方式：
cv2.INTER_AREA:使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN方法　　　　
cv2.INTER_CUBIC: 立方插值
cv2.INTER_LINEAR: 双线形插值　
cv2.INTER_NN: 最近邻插值
```


### 3.1.4 保存
* numpy.save, numpy.savez
* h5py

## 3.2 tensorflow


## 3.3 pytorch
