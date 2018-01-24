# 一. 软件安装
linux，ubuntu</br>
vs2015（不需要）</br>
驱动 [cuda](https://developer.nvidia.com/cuda-toolkit-archive), cudnn60</br>
git, anaconda</br>
python, java</br>
更改conda, pip镜像</br>
tensorflow-gpu, pytorch, keras</br>
xgboost：git, mingw or tdm-gcc</br>

pymc3, edward, pystan</br>
[lingo](https://zhuanlan.zhihu.com/p/29772798), pulp</br>

word, excel, ppt</br>
bdp</br>

everything，tampermonkey</br>
sharelatex，processon</br>

# 二. python及爬虫
## 2.1 python知识
1. 显示进度条
``` java
from tqdm import tqdm
for i in tqdm(range(100))
```
2. 覆盖打印 
``` java
from time import sleep
print（"\r", object, end = ""）
```
3. 格式化输出
 ``` java
>>> print('{} {}'.format('hello','world'))  # 不带字段, 默认左对齐
hello world
>>> print('{0} {1} {0}'.format('hello','world'))  # 打乱顺序
hello world hello
>>> print('{a} {tom} {a}'.format(tom='hello',a='world'))  # 带关键字
world hello world
>>> print('{:10s} and {:>10s}'.format('hello','world'))  # 取10位左对齐，取10位右对齐
hello      and      world
>>> print('{:^10s} and {:^10s}'.format('hello','world'))  # 取10位中间对齐
  hello    and   world   
>>> print('{} is {:.2f}'.format(1.123,1.123))  # 取2位小数
1.123 is 1.12
>>> print('{0} is {0:>10.2f}'.format(1.123))  # 取2位小数，右对齐，取10位
1.123 is       1.12
```
## 2.2 常用模块
1. os模块 </br>
os.remove() 删除文件
os.rename() 重命名文件
os.walk() 生成目录树下的所有文件名
os.chdir() 改变目录
os.mkdir/makedirs 创建目录/多层目录
os.rmdir/removedirs 删除目录/多层目录
os.listdir() 列出指定目录的文件
os.getcwd() 取得当前工作目录
os.chmod() 改变目录权限
os.path.basename() 去掉目录路径，返回文件名
os.path.dirname() 去掉文件名，返回目录路径
os.path.join() 将分离的各部分组合成一个路径名
os.path.split() 返回( dirname(), basename())元组
os.path.splitext() 返回 (filename, extension) 元组
os.path.getatime\ctime\mtime 分别返回最近访问、创建、修改时间
os.path.getsize() 返回文件大小
os.path.exists() 是否存在
os.path.isabs() 是否为绝对路径
os.path.isdir() 是否为目录
os.path.isfile() 是否为文件

* os.walk(path),遍历path，返回一个对象，他的每个部分都是一个三元组,('目录x'，[目录x下的目录list]，目录x下面的文件)
``` python
import os
def walk_dir(dir,fileinfo,topdown=True):
    for root, dirs, files in os.walk(dir, topdown):
        for name in files:
            print(os.path.join(name))
            fileinfo.write(os.path.join(root,name) + '\n')
        for name in dirs:
            print(os.path.join(name))
            fileinfo.write('  ' + os.path.join(root,name) + '\n')
dir = raw_input('please input the path:')
fileinfo = open('list.txt','w')
walk_dir(dir,fileinfo)
```
2. argparse模块
``` python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                    help="increase output verbosity", action='store_true', default=True)
args = parser.parse_args()
```
3. collections模块
* counter
``` python
>>> from collections import Counter
>>> c = Counter()
>>> for ch in 'programming':
        c[ch] = c[ch] + 1
>>> c
Counter({'g': 2, 'm': 2, 'r': 2, 'a': 1, 'i': 1, 'o': 1, 'n': 1, 'p': 1})
```
* Orderdict:使用dict时，Key是无序的。在对dict做迭代时，我们无法确定Key的顺序,如果要保持Key的顺序，可以用OrderedDict：
``` python
>>> from collections import OrderedDict
>>> d = dict([('a', 1), ('b', 2), ('c', 3)])
>>> d # dict的Key是无序的
{'a': 1, 'c': 3, 'b': 2}
>>> od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
>>> od # OrderedDict的Key是有序的
OrderedDict([('a', 1), ('b', 2), ('c', 3)])
```
4. sys模块







# 三. 数据读取与保存
## 3.1 python读取
### 3.1.1 txt, csv, excel等文件
1. txt
``` python
f.readline() #读取一行
f.readlines() ## 返回每一行所组成的列表

f = codecs.open('', 'r'， encoding='')
for line in f: ## 迭代读取大文件
```
2. csv和excel
``` python
f = pd.read_csv(, sep=)
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
# 关键字参数为dst,fx,fy,interpolation
# dst为缩放后的图像
# dsize为(w,h),但是image是(h,w,c)
# fx,fy为图像x,y方向的缩放比例，
# interplolation为缩放时的插值方式，有三种插值方式：
# cv2.INTER_AREA:使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN方法　　　　
# cv2.INTER_CUBIC: 立方插值
# cv2.INTER_LINEAR: 双线形插值　
# cv2.INTER_NN: 最近邻插值
```

### 3.1.4 保存
* numpy.save, numpy.savez, numpy.load
* h5py
``` python
import h5py
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X_train', data=X)
h5f.create_dataset('y_train', data=y)
h5f.close()

h5f = h5py.File('data.h5', 'r')
X = h5f['X_train']
Y = h5f['y_train']
h5f.close()
```
### 3.1.2 文件写入


## 3.2 tensorflow

## 3.3 pytorch

## 3.4 keras


# 四. 结合论文熟悉不同框架
## 4.1 MRC（机器阅读理解）
### 4.1.1 r-net
### 4.1.2 FusionNet
### 4.1.3 DCN+

## 4.2 VQA（视觉问答）
### 4.2.1 MCB
### 4.2.2 MFB
### 4.2.3 MUTAN, Tucker Decomposition
### 4.2.4 第一名方案

## 4.3 细粒度图像识别

## 4.4 Knowledge graph（知识图谱）

## 4.5 知识库问答

## 4.6 GAN

## 4.7 强化学习

# 五. 不同框架知识
## 5.1 sklearn
1. 绘制confusion matrix
``` python
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
```
## 5.2 xgboost [英文](http://xgboost.readthedocs.io/en/latest/), [中文](http://xgboost.apachecn.org/cn/latest/)
## 5.3 tensorflow [中文](https://tensorflow.google.cn/)
## 5.4 keras [中文](http://keras-cn.readthedocs.io/en/latest/)
### 5.4.1 不同模型权重
1. (https://github.com/freelzy/Baidu_Dogs)</br>
2. [Densenet](https://github.com/flyyufelix/DenseNet-Keras)
### 5.4.2
1. 回调函数
from keras.callbacks import EarlyStopping  
from keras.callbacks import TensorBoard
early_stopping =EarlyStopping(monitor='val_loss', patience=20) 
tb = Tensorboard(log_dir='./log')

## 5.5 pytorch  [英文](http://pytorch.org/), [中文](http://pytorch.apachecn.org/cn/0.3.0/)
### 5.5.1 不同模型权重
(https://github.com/Cadene/pretrained-models.pytorch)


