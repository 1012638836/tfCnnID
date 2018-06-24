#coding=utf-8
#数据拆分
import math 
import os
import numpy as np 

#导入 tensorflow 
import tensorflow as tf

"""
 数据 打标签 并拆分成 测试和训练数据

"""
# 0 类 和 0类 标签
zeroclass = []   
label_zeroclass = []
# 1类 和  1类标签
oneclass=[]   
label_oneclass=[]
# 2类 和 2类标签
twoclass=[]
label_twoclass=[]
#3类 和 3类标签
threeclass=[]
label_threeclass=[]
#4类 和 4类标签
fourclass=[]
label_fourclass=[]
#5类 和 5类标签
fiveclass=[]
label_fiveclass=[]
#6类 和 6类标签
sixclass=[]
lable_sixclass=[]
#7类 和 7类标签
sevenclass=[]
lable_sevenclass=[]
#8类 和 8类标签
eightclass=[]
label_eightclass=[]
#9类 和 9类标签
nineclass=[]
lable_nineclass=[]

def get_files(file_dir,ratio):  
    for file in os.listdir(file_dir+'/0'):    
        zeroclass.append(file_dir +'/0'+'/'+ file)     
        label_zeroclass.append(0)    
    for file in os.listdir(file_dir+'/1'):    
        oneclass.append(file_dir +'/1'+'/'+file)    
        label_oneclass.append(1)    
    for file in os.listdir(file_dir+'/2'):    
        twoclass.append(file_dir +'/2'+'/'+ file)     
        label_twoclass.append(2)    
    for file in os.listdir(file_dir+'/3'):    
        threeclass.append(file_dir +'/3'+'/'+file)    
        label_threeclass.append(3)        
    for file in os.listdir(file_dir+'/4'):    
        fourclass.append(file_dir +'/4'+'/'+file)    
        label_fourclass.append(4)        
    for file in os.listdir(file_dir+'/5'):    
        fiveclass.append(file_dir +'/5'+'/'+file)    
        label_fiveclass.append(5)
    for file in os.listdir(file_dir+'/6'):    
        sixclass.append(file_dir +'/6'+'/'+file)    
        lable_sixclass.append(6)
    for file in os.listdir(file_dir+'/7'):    
        sevenclass.append(file_dir +'/7'+'/'+file)    
        lable_sevenclass.append(7)
    for file in os.listdir(file_dir+'/8'):    
        eightclass.append(file_dir +'/8'+'/'+file)    
        label_eightclass.append(8)  

    for file in os.listdir(file_dir+'/9'):    
        nineclass.append(file_dir +'/9'+'/'+file)    
        lable_nineclass.append(9)  

    ##对生成图片路径和标签list打乱处理（img和label）  
    image_list=np.hstack((zeroclass, oneclass, twoclass, threeclass, fourclass, fiveclass,sixclass,sevenclass,eightclass,nineclass))  
    label_list=np.hstack((label_zeroclass, label_oneclass, label_twoclass, label_threeclass, label_fourclass,
    label_fiveclass,lable_sixclass,lable_sevenclass,label_eightclass,lable_nineclass))  

    #shuffle打乱  
    temp = np.array([image_list, label_list])  
    temp = temp.transpose()  
    np.random.shuffle(temp)  

    #将所有的img和lab转换成list  
    all_image_list=list(temp[:,0])  
    all_label_list=list(temp[:,1])  

    #将所得List分为2部分，一部分train,一部分val，ratio是验证集比例  
    n_sample = len(all_label_list)    
    n_val = int(math.ceil(n_sample*ratio))   #验证样本数    
    n_train = n_sample - n_val   #训练样本数    
    
    tra_images = all_image_list[0:n_train]  
    tra_labels = all_label_list[0:n_train]    
    tra_labels = [int(float(i)) for i in tra_labels]    
    val_images = all_image_list[n_train:]    
    val_labels = all_label_list[n_train:]  
    val_labels = [int(float(i)) for i in val_labels]      
    return tra_images,tra_labels,val_images,val_labels  



"""
将图片转为 tensorFlow 能读取的张量
"""
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #数据转换
    image = tf.cast(image, tf.string)   #将image数据转换为string类型
    label = tf.cast(label, tf.int32)    #将label数据转换为int类型
    #入队列
    input_queue = tf.train.slice_input_producer([image, label])
    #取队列标签 张量
    label = input_queue[1] 
    #取队列图片 张量
    image_contents = tf.read_file(input_queue[0])

    #解码图像，解码为一个张量
    image = tf.image.decode_jpeg(image_contents, channels=3)

    #对图像的大小进行调整，调整大小为image_W,image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #对图像进行标准化
    image = tf.image.per_image_standardization(image)

    #等待出队
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size]) #将label_batch转换格式为[]
    image_batch = tf.cast(image_batch, tf.float32)   #将图像格式转换为float32类型
  
    return image_batch, label_batch  #返回所处理得到的图像batch和标签batch

