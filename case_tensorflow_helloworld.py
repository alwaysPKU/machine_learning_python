import tensorflow as tf
import numpy as np


# 初始化一个Tensorflow的常量： hello google Tensorflow! 字符串，并命名为greeting作为一个计算模块
greeting = tf.constant('hello google Tensorflow')

# 启动一个会话，
sess = tf.Session()
# 使用会话执行greeting计算模块
result = sess.run(greeting)
# 输出会话结果的执行
print(result)
# 关闭会话
sess.close()

# 线性计算描述怎么用会话计算的
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
linear = tf.add(product, tf.constant(2.0))

with tf.Session() as sess:
    result = sess.run(linear)
    print(result)
