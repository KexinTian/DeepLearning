
import os
import numpy as np


""""
生成casiadatabase_data.txt，包含整个数据库的数据
在这之前，需要已经把每个人的语音特征都提取出来了
"""
loc="F:\\my_datasets\\casiadatabase"
sub_dirs=os.listdir(loc)
data_list=[]
data_num=0
for sub_dir in sub_dirs:
    sub_dir_path=os.path.join(loc,sub_dir)
    if os.path.isdir(sub_dir_path):
        files = os.listdir(
            sub_dir_path
        )
        if files.__contains__("data.txt"):
            this_feature = np.loadtxt(os.path.join(sub_dir_path, "data.txt"), skiprows=0)
            data_list.append(this_feature)
            data_num=data_num+len(this_feature)
data=np.array(data_list)
data=data.reshape(data_num,-1)
print(data.shape)
print (data[1][0:5])
loc = os.path.join(loc, "casiadatabase_data.txt")
np.savetxt(loc, data)#如果txt存在，则重写；如果没有，则创建

