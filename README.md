# 交通事故预测
该项目的主要任务是对纽约市的交通事故进行预测
## 环境安装
在python3.6环境下使用如下pip命令安装所需包
* pip install torch
* pip install tqdm
* pip install matplotlib
## 数据
可以直接使用已经处理好的数据，只需解压即可
```
unzip data.zip -d ./data
```
解压后可使用如下命令观察数据：
```
python observe.py
```
本次实验所使用到的数据分别为：
* [Motor Vehicle Collisions - Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
* [Yellow Taxi Trips](https://data.cityofnewyork.us/browse?Dataset-Information_Agency=Taxi+and+Limousine+Commission+%28TLC%29&)
* [Green Taxi Trips](https://data.cityofnewyork.us/browse?Dataset-Information_Agency=Taxi+and+Limousine+Commission+%28TLC%29&)
* [Uber](https://github.com/fivethirtyeight/uber-tlc-foil-response)
## 训练
```
python main.py
```
训练模型会存储在experiments文件夹下。

实验配置可以在`config.py`文件中进行更改，这里对其中几个重要参数进行说明
1. granularity：时间粒度。取值为day或hour，表示以天或者小时为单位进行训练；
2. year：数据年份。可取值为2014，2015或all，all表示使用从2014年到2021年的全部数据；
3. add_traffic：是否加入traffic volume作为一个特征，取值为True/False。如果是，则会在year=2014时加入taxi traffic以及Uber traffic进行训练，在year=2015时加入taxi traffic进行训练。
## 测试
```
python main.py -mode test -path ${your_exp_path}
```
`${your_exp_path}`表示训练后模型存储的路径。测试完成后不仅会输出模型在测试集的结果，还会在模型存储的路径下保存预测样例图片。
