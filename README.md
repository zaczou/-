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
[bdp](https://me.bdp.cn/home.html)</br>

everything, tampermonkey</br>
[sharelatex](https://www.sharelatex.com/), [processon](https://www.processon.com/)</br>

jupyter notebook, [colabotorary](https://zhuanlan.zhihu.com/p/33125415) [知乎](https://zhuanlan.zhihu.com/p/33232118)

[jijidown](http://client.jijidown.com/)

# 二. python及爬虫
## 2.1 python知识
1. 显示进度条
``` python
from tqdm import tqdm
for i in tqdm(range(100))
```
2. 覆盖打印 
``` python
from time import sleep
print（"\r", object, end = ""）
sleep(1)
```
3. 格式化输出
 ``` python
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
``` python
os.remove() # 删除文件
os.rename() # 重命名文件
os.walk() # 生成目录树下的所有文件名
os.chdir() # 改变目录
os.mkdir/makedirs # 创建目录/多层目录
os.rmdir/removedirs # 删除目录/多层目录
os.listdir() # 列出指定目录的文件
os.getcwd() # 取得当前工作目录
os.chmod() # 改变目录权限
os.path.basename() # 去掉目录路径，返回文件名
os.path.dirname() # 去掉文件名，返回目录路径
os.path.join() # 将分离的各部分组合成一个路径名
os.path.split() # 返回(dirname(), basename())元组
os.path.splitext() # 返回 (filename, extension) 元组
os.path.getatime,ctime,mtime # 分别返回最近访问、创建、修改时间
os.path.getsize() # 返回文件大小
os.path.exists() # 是否存在
os.path.isabs() # 是否为绝对路径
os.path.isdir() # 是否为目录
os.path.isfile() # 是否为文件
```
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
dir = input('please input the path:')
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
``` python
sys.argv # 命令行参数List，第一个元素是程序本身路径
sys.modules.keys() # 返回所有已经导入的模块列表
sys.exc_info() # 获取当前正在处理的异常类,exc_type、exc_value、exc_traceback当前处理的异常详细信息
sys.exit(n) # 退出程序，正常退出时exit(0)
sys.hexversion # 获取Python解释程序的版本值，16进制格式如：0x020403F0
sys.version # 获取Python解释程序的版本信息
sys.maxint # 最大的Int值
sys.maxunicode # 最大的Unicode值
sys.modules # 返回系统导入的模块字段，key是模块名，value是模块
sys.path # 返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
sys.platform # 返回操作系统平台名称
sys.stdout # 标准输出
sys.stdin # 标准输入
sys.stderr # 错误输出
sys.exc_clear() # 用来清除当前线程所出现的当前的或最近的错误信息
sys.exec_prefix # 返回平台独立的python文件安装的位置
sys.byteorder # 本地字节规则的指示器，big-endian平台的值是'big',little-endian平台的值是'little'
sys.copyright # 记录python版权相关的东西
sys.api_version # 解释器的C的API版本
```

5. re模块

## 2.3 常用数据处理模块
1. numpy

2. pandas

3. matplotlib


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
[基础](https://github.com/zhedongzheng/finch)
## 4.1 MRC（机器阅读理解）
* [中文阅读理解比赛](https://mp.weixin.qq.com/s/vAj7vUkvPS7jqHzewb5AuQ)
* [squad](https://rajpurkar.github.io/SQuAD-explorer/)
* [ms marco](http://www.msmarco.org/leaders.aspx)
* [trivalqa](https://competitions.codalab.org/competitions/17208#learn_the_details)
### 4.1.1 r-net
* [github](https://github.com/search?l=Python&o=desc&q=r-net&s=&type=Repositories&utf8=%E2%9C%93)
* [tensorflow](https://github.com/HKUST-KnowComp/R-Net)
* [pytorch](https://github.com/matthew-z/R-net)
* [keras](https://github.com/YerevaNN/R-NET-in-Keras)
### 4.1.2 FusionNet
* [tensorflow](https://github.com/obryanlouis/qa)
* [pytorch](https://github.com/exe1023/FusionNet)
### 4.1.3 DCN+

### 4.1.4 Simple and effective multi-paragraph reading comprehension
* [tensorflow](https://github.com/allenai/document-qa)

## 4.2 VQA（视觉问答）
* [比赛](http://visualqa.org/roe_2017.html)
### 4.2.1 [MCB]
### 4.2.2 [MFB]
* [pytorch](https://github.com/asdf0982/vqa-mfb.pytorch)
* [caffe-py](https://github.com/yuzcccc/vqa-mfb)
### 4.2.3 [MUTAN, Tucker Decomposition]
* [pytorch](https://github.com/cadene/vqa.pytorch)
### 4.2.4 2017 第一名方案
* [pytorch](https://github.com/markdtw/vqa-winner-cvprw-2017) 
### 4.2.5 bottom-up-attention-vqa
* [pytorch](https://github.com/hengyuan-hu/bottom-up-attention-vqa)
## 4.3 细粒度图像识别

## 4.4 Knowledge graph（知识图谱）
* [复旦知识工厂](http://kw.fudan.edu.cn/#projects)

## 4.5 知识库问答

## 4.6 GAN

## 4.7 强化学习

## 4.8 机器翻译
### 4.8.1 attention is all you need
[github](https://github.com/search?l=Python&q=attention+is+all+you+need&type=Repositories&utf8=%E2%9C%93)
* multi head attention
[tensorflow](https://github.com/bojone/attention/blob/master/attention_tf.py)
[pytorch]
[keras](https://github.com/bojone/attention/blob/master/attention_keras.py)









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
* [常用](https://github.com/lizhaoliu/tensorflow_snippets)


## 5.4 keras [中文](http://keras-cn.readthedocs.io/en/latest/)
### 5.4.1 不同模型权重
1. (https://github.com/freelzy/Baidu_Dogs)</br>
2. [Densenet](https://github.com/flyyufelix/DenseNet-Keras)
### 5.4.2 常用层或函数
1. 自定义层
* 简单Lambda, 无需参数训练
* 参数训练
``` python
class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3])
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
```

2. 回调函数
```
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint 
early_stopping = EarlyStopping(monitor='val_loss', patience=20) 
tb = Tensorboard(log_dir='./log')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto')
model.fit(, callbacks=[early_stopping, tb, checkpoint])
```
3. [keras attention block](https://github.com/NLP-Deeplearning-Club/keras_attention_block)


## 5.5 pytorch  [英文](http://pytorch.org/), [中文](http://pytorch.apachecn.org/cn/0.3.0/)
### 5.5.1 不同模型权重
(https://github.com/Cadene/pretrained-models.pytorch)


