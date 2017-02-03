# Neural-Style-tf
基于Tensoflow的风格迁移实例。

## 依赖
- tensorflow
- numpy
- scipy

## 效果
Contents: <br/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/images/content_0.jpg" width="30%"/> 
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/images/content_1.jpg" width="30%"/> 
<br/>
Styles: <br/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/images/style_0.jpg" width="30%"/> 
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/images/style_1.jpg" width="30%"/> 
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/images/style_2.jpg" width="30%"/> 
<br/>
Results: <br/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/results/result_00.jpg" width="30%"/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/results/result_01.jpg" width="30%"/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/results/result_02.jpg" width="30%"/>
<br/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/results/result_10.jpg" width="30%"/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/results/result_11.jpg" width="30%"/>
<img src="https://github.com/PengZiqiao/Neural-Style-tf/blob/master/results/result_12.jpg" width="30%"/>
<br/>

## 常量
- CONTENT_IMG：content图片地址+文件名
- STYLE_IMG：style图片地址+文件名
- OUTOUT_DIR： result图片地址
- OUTPUT_IMG：results图片文件名
- VGG_MODEL： imagenet-vgg-verydeep-19.mat地址+文件名。这个文件太大了，就不上传了
- INI_NOISE_RATIO：梯度下降初始图片由随机噪音和content合成，这里设置噪音占比
- STYLE_WEIGHT：style占的比重
- ITERATION：迭代次数
