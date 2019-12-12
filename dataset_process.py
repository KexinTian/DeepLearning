import pydub
import io
import wave
import os
import sys
from utils import utils
from time import gmtime
from time import strftime
import subprocess
from absl import flags, app
import numpy as np
from sklearn import preprocessing as skpp
import  pandas as pd
from pandas import DataFrame
flags.DEFINE_boolean("need_MP3_to_WAV","False","是否需要将MP3转化为WAV")
flags.DEFINE_boolean("need_feature_SMILExtract","False","是否需要提取语音特征")
flags.DEFINE_boolean("need_save_features","True","是否需要保存语音特征集")
flags.DEFINE_string("mp3_dir", "", "保存mp3的文件夹")
flags.DEFINE_string("wav_dir", "F:/my_datasets/casiadatabase/wangzhe","用于保存wav的文件夹")
flags.DEFINE_string("feature_dir", "F:/my_datasets/casiadatabase/wangzhe","用于保存语音提取出来的文特征")
flags.DEFINE_integer("num_classes", "6","情感类别数目")
flags.DEFINE_integer("num_features", "384","情感特征维数")
FLAGS = flags.FLAGS

class Log(object):

    def __init__(self, flag, des='',obj=None):
        self.flag = flag
        self.des = des
        self.obj=obj

    def set_flag(self,flag):
        self.flag = flag

    def set_des(self, des):
        self.des = des

    def show(self):
        print("{0} : {1}\r\n".format(self.flag, self.des))


def MP3_to_WAV_1(mp3_path,wav_path):
    """
    这是MP3文件转化成WAV文件的函数
    :param mp3_path是MP3文件
    :param ,wav_path是转化后的文件
    :return: bool类型，成功返回True，失败返回False
    """
    try:
        with open(mp3_path, 'rb') as fh:
            data = fh.read()

        aud = io.BytesIO(data)
        pydub.AudioSegment.converter = "D://NoSpaceInstall//ffmpeg//bin//ffmpeg.exe"
        sound = pydub.AudioSegment.from_file(aud, format='mp3')
        raw_data = sound._data

        size = len(raw_data)
        f = wave.open(wav_path, 'wb')
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.setnframes(size)
        f.writeframes(raw_data)
        f.close()
        log = Log(True, "MP3_to_WAV_1方法正确")
        return log
    except Exception as e:
        log = Log(False, "MP3_to_WAV_1方法出错, " +repr(e))
        log.show()
        return log



def MP3_to_WAV_2(mp3_path,wav_path):
    try:
        pydub.AudioSegment.converter ="D://NoSpaceInstall//ffmpeg//bin//ffmpeg.exe"
        MP3_File = pydub.AudioSegment.from_mp3(file=mp3_path)
        MP3_File.export(wav_path, format="wav")
        log = Log(True, "MP3_to_WAV_2方法正确")
        return log
    except Exception as e:
        log = Log(False, "MP3_to_WAV_2方法出错, " +repr(e))
        log.show()
        return log


def batchpro_MP3_to_WAV(mp3_dir,wav_dir):
    """
    批量处理，将MP3文件转化成WAV文件
    :param mp3_dir是MP3保存的文件夹
    :param ,wav_dir是转化后wav保存的文件夹
    :return: 成功返回True，失败返回False
    """
    try:
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
            print("转化后wav保存在: {0}".format(wav_dir))
        if os.path.isdir(mp3_dir):
            mp3List = os.listdir(mp3_dir)
            if len(mp3List)==0:
                log=Log(False,"批量处理MP3_to_WAV时，MP3文件夹为空")
                log.show()
            for mp3 in mp3List:
                if not mp3[-4:]=='.wav':
                    continue
                mp3_base_path=os.path.abspath(mp3)
                #mp3_base_path=os.path.join(os.getcwd(), mp3)
                #print("mp3_base_path: {0}".format(mp3_base_path))
                wav_path=mp3.split('.')[0]+'.wav'
                wav_base_path=os.path.join(wav_dir, wav_path)
                log=MP3_to_WAV_1(mp3_base_path,wav_base_path)
                if log.flag==False:
                    break
        if log.flag==True:
            log.set_des("批量处理MP3_to_WAV完成")
            log.show()
        return log
    except Exception as e:
        log = Log(False, "批量处理MP3_to_WAV时出错, "+repr(e))
        log.show()
        return log



def feature_SMILExtract(wav_dir,feature_dir):
    """
    opensmile提取wav_dir 中所有wav的特征，并保存在feature_dir中.单条语音的特征采用np.savetxt保存，类型为float
    :param wav_dir: 
    :param feature_dir: 
    :return: Log对象
    """
    try:
        if not os.path.isdir(wav_dir) :
            log = Log(False, "feature_SMILExtract方法出错，"+wav_dir+"不是文件夹")
            log.show()
            return log
        if not os.path.isdir(feature_dir):
            os.makedirs(feature_dir)
        wavList = os.listdir(wav_dir)
        if len(wavList) == 0:
            log = Log(False, "feature_SMILExtract方法出错，"+wav_dir+"文件夹为空")
            log.show()
            return log
        for wav in wavList:
            if wav[-4:] == '.wav':
                this_path_input = os.path.join(wav_dir, wav)
                this_path_output = os.path.join(feature_dir, wav[:-4] + '.txt')
                f=open(this_path_output,'w')
                f.close()
                cmd = 'cd /d D:/NoSpaceInstall/opensmile-2.3.0/bin/Win32 && SMILExtract_Release -C D:/NoSpaceInstall/opensmile-2.3.0/config/IS09_emotion.conf -I ' + this_path_input + ' -O ' + this_path_output
                ret = subprocess.call(cmd, shell=True)
                f = open(this_path_output)
                last_line = f.readlines()[0]
                f.close()
                os.remove(this_path_output)
                features = last_line.split(',')
                ###去掉第一个和最后一个元素
                features = features[1:-1]
                features_array=np.array(features,dtype=float)
                np.savetxt(this_path_output,features_array)
                log = Log(True, "单条语音特征化简完毕")
                # os.system(cmd)
        log = Log(True, "feature_SMILExtract完成")
        log.show()
        return log
    except Exception as e:
        print(repr(e))
        log = Log(False, "feature_SMILExtract出错, "+repr(e))
        log.show()
        return log



def main(argv):
  del argv
  filename = os.path.join(
        os.getcwd(), "log-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".txt")
  sys.stdout = utils.Logger(filename)#原来sys.stdout指向控制台，现在重定向到文件

  if FLAGS.need_MP3_to_WAV:
      mp3_dir=FLAGS.mps_dir
      wav_dir=FLAGS.wav_dir
      batchpro_MP3_to_WAV(mp3_dir, wav_dir)

  if FLAGS.need_feature_SMILExtract:
      """
      需要提取特征（文件夹之间的逻辑关系要根据实际情况改写）
      """
      wav_dir=FLAGS.wav_dir
      file_list=os.listdir(wav_dir)
      for f in file_list:
          sub_wav_dir = os.path.join(wav_dir, f)
          if  os.path.isdir(sub_wav_dir):#这个函数只能判断绝对路径
              feature_dir=os.path.join(wav_dir, f+"_feature")
              feature_SMILExtract(sub_wav_dir, feature_dir)#sub_wav_dir中不能有文件夹

  if FLAGS.need_save_features:
      """
      需要把特征都整合到一个文件里。（文件夹之间的逻辑关系要根据实际情况改写）
      """
      personal_dir = FLAGS.feature_dir
      file_list = os.listdir(personal_dir)
      feature_list = []
      y_list = []
      for t in file_list:
          if t.__contains__("angry"):
              y = 0
          elif t.__contains__("fear"):
              y = 1
          elif t.__contains__("happy"):
              y = 2
          elif t.__contains__("neutral"):
              y = 3
          elif t.__contains__("sad"):
              y = 4
          else:  # surprise
              y = 5
          sub_dir = os.path.join(personal_dir, t)
          if os.path.isdir(sub_dir):  # 这个函数只能判断绝对路径
              for f in os.listdir(sub_dir):
                  if f[-4:] == ".txt":
                      this_feature_path = os.path.join(sub_dir, f)
                      this_feature=np.loadtxt(this_feature_path,skiprows=0)
                      feature_list.append(this_feature)
                      y_list.append(y)
      if len(y_list) != 0:
          y_array = np.array(y_list)
          y_array = y_array.reshape(len(y_array), 1)

          feature_array = np.array(feature_list)
          feature_array = feature_array.reshape(len(feature_array), -1)

          data = np.column_stack((y_array, feature_array))
          print("{0} 中的语音情感特征整合完成，保存在data.txt中,数据集大小{1}\n".format(personal_dir,data.shape))

          loc = os.path.join(personal_dir,personal_dir.split('/')[-1]+"_data.txt")
          np.savetxt(loc, data)#如果txt存在，则重写；如果没有，则创建




if __name__ == "__main__":
  app.run(main)




