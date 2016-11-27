##BaseLine
###数据集
包括了5,590 LFW 数据集中的图片 和 7,876 张来自网络的图片, 可以在[这里](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip)下载，将解压后的文件放入**dataset/train**文件夹下。
###数据生成
**generate.py**用于将数据集中jpg格式的文件转成hdf5格式的数据，放入train文件夹下。运行python generate.py生成数据。common文件夹下是一些必要函数的定义，用于图像处理。
###训练
**prototext**文件夹下是网络结构(含有1_F, 1_EN, 1_NM三种结构)的定义, 运行脚本sh train.sh训练三层网络，训练好的结果存放在model文件夹下
###测试
运行脚本sh test.sh进行网络测试，测试结果存放在log文件夹下，测试程序放在test文件夹下，分别进行整体网络测试与局部网络测试。
###清空结果
运行脚本sh clear.sh，清空所有训练好的model以及生成的数据，可进行下一次训练。
###参考文献
[1. Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
