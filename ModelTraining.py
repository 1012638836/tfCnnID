#coding=utf-8
import os
import numpy as np
import tensorflow as tf

import DataUtils
import MainModel


from PIL import Image  
import matplotlib.pyplot as plt 

N_CLASSES = 10 # 10分类
capacity=256   #队列容量
BATCH_SIZE=10
MAX_STEP = 2000 #最大训练步骤
IMG_W = 28 #图片的宽度
IMG_H = 28 #图片的高度
learning_rate = 0.0001  #学习率

"""
 定义开始训练的函数
"""
def run_training():
    """
    ##1.数据的处理
    """
    # 训练图片路径
    train_dir = '/home/zhang-rong/Yes/CnnID/train/'
    # 输出log的位置
    logs_train_dir = '/home/zhang-rong/Yes/CnnID/log/'

    # 模型输出
    train_model_dir = '/home/zhang-rong/Yes/CnnID/model/'

    tra_list,tra_labels,val_list,val_labels=DataUtils.get_files(train_dir,0.2)
    tra_list_batch,tra_label_batch=DataUtils.get_batch(tra_list,tra_labels,IMG_W,IMG_H,BATCH_SIZE,capacity) # 转成tensorflow 能读取的格式的数据
    val_list_batch,val_label_batch=DataUtils.get_batch(val_list,val_labels,IMG_W,IMG_H,BATCH_SIZE,capacity)
    print val_list,"******",val_labels

    """
    ##2.网络的推理
    """
    # 进行前向训练，获得回归值
    train_logits = MainModel.inference(tra_list_batch, BATCH_SIZE, N_CLASSES)

    """
    ##3.定义交叉熵和 要使用的梯度下降的 优化器 
    """
    # 计算获得损失值loss
    train_loss = MainModel.losses(train_logits, tra_label_batch)
    # 对损失值进行优化
    train_op = MainModel.trainning(train_loss, learning_rate)

    """
    ##4.定义后面要使用的变量
    """
    # 根据计算得到的损失值，计算出分类准确率
    train__acc = MainModel.evaluation(train_logits, tra_label_batch)
    # 将图形、训练过程合并在一起
    summary_op = tf.summary.merge_all()

    # 新建会话
    sess = tf.Session()
  
    # 将训练日志写入到logs_train_dir的文件夹内
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()  # 保存变量

    # 执行训练过程，初始化变量
    sess.run(tf.global_variables_initializer())

    # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    coord = tf.train.Coordinator()
    # 启动入队的线程，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    """
    进行训练：
    使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，
    会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;
    """
 
    try:
        for step in np.arange(MAX_STEP): #从0 到 2000 次 循环
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc]) 


            # 每50步打印一次损失值和准确率
            if step % 2 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)



    # 如果读取到文件队列末尾会抛出此异常
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()       # 使用coord.request_stop()来发出终止所有线程的命令

    coord.join(threads)            # coord.join(threads)把线程加入主线程，等待threads结束


    checkpoint_path = os.path.join(train_model_dir, 'model.ckpt')
    
    # saver.save(sess, checkpoint_path, global_step=step)
    
    saver.save(sess, checkpoint_path)
    sess.close()                   # 关闭会话


def get_one_image_file(img_dir):
    
    image = Image.open(img_dir)
    plt.legend()
    plt.imshow(image)   #显示图片
    image = image.resize([28, 28])
    image = np.array(image)
    return image


"""
进行单张图片的测试
"""
def evaluate_one_image():

    image_array=get_one_image_file("/home/zhang-rong/Yes/CnnID/test_yes/8.jpg")

    with tf.Graph().as_default():
        BATCH_SIZE = 1   # 获取一张图片
        N_CLASSES = 10  #10分类

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 28, 28, 3])     #inference输入数据需要是4维数据，需要对image进行resize
        logit = MainModel.inference(image, BATCH_SIZE, N_CLASSES)       
        logit = tf.nn.softmax(logit)    #inference的softmax层没有激活函数，这里增加激活函数

        #因为只有一副图，数据量小，所以用placeholder
        x = tf.placeholder(tf.float32, shape=[28, 28, 3])

        # 
        # 训练模型路径
        logs_train_dir = '/home/zhang-rong/Yes/CnnID/model/'

       
        saver=tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess,str(logs_train_dir+"model.ckpt"))

            prediction = sess.run(logit, feed_dict={x: image_array})
            # 得到概率最大的索引
            max_index = np.argmax(prediction)
            print "识别出来的身份证数字为：",max_index

                

"""
主函数
"""
def main():
    # run_training()
    evaluate_one_image()


if __name__ == '__main__':
    main()