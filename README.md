# 一. 软件安装
linux，ubuntu</br>
vs2015（不需要）</br>
驱动 cuda8.0 cudnn60</br>
git, anaconda</br>
python, java</br>
更改conda, pip镜像</br>
tensorflow-gpu, pytorch, keras</br>
xgboost：git, mingw or tdm-gcc</br>

pymc3, edward, pystan</br>
lingo, pulp</br>

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
image = Image.open('test.jpg')
image = np.array(image,dtype=np.float32) # image = np.array(image)认是uint8
```

2. Matplotlib
3. skimage



### 3.1.4 保存
* numpy.save, numpy.savez
* h5py

## 3.2 tensorflow


## 3.3 pytorch
