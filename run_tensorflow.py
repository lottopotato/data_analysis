import tensorflow as tf
import numpy as np

from plot import *
from numpy_process import sampling_arr, list2numpy
from data_type import dataType

class autoEncoder:
     def __init__(self, data, test = 100, learning_rate = 0.01, step = 1, print_step = 10):
          self.data = data
          self.dataAt = dataType()
          self.data_row = int( self.data.shape[0] / self.dataAt.SRC_len )
          self.dataMax = self.data.max()
          self.test_space = test
          
          self.learning_rate = learning_rate
          self.num_steps = step
          self.print_point = print_step
          
          self.num_hidden_1 = 256
          self.num_hidden_2 = 128
          self.num_input = self.dataAt.SRC_len

          self.X = tf.placeholder("float", [1, self.num_input])

          self.weights = {
               "encoder_h1" : tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
               "encoder_h2" : tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
               "decoder_h1" : tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
               "decoder_h2" : tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input]))
               }
          self.biases = {
               "encoder_b1" : tf.Variable(tf.random_normal([self.num_hidden_1])),
               "encoder_b2" : tf.Variable(tf.random_normal([self.num_hidden_2])),
               "decoder_b1" : tf.Variable(tf.random_normal([self.num_hidden_1])),
               "decoder_b2" : tf.Variable(tf.random_normal([self.num_input]))
               }
          
     def printOption(self):
          print(" ========================== ")
          print(" auto encoder ")
          print(" learning rate : " + str(self.learning_rate))
          print(" train step : " + str(self.num_steps))
          print(" data row length : " + str(self.data_row))
          print(" train row length : " + str(self.data_row - self.test_space))
          print(" test row length : " + str(self.test_space))
          print(" ========================== ")
          
     def encoder(self, x):
          layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                              self.biases['encoder_b1']))
          layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                              self.biases['encoder_b2']))
          return layer_2

     def decoder(self, x):
          layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                              self.biases['decoder_b1']))
          layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                              self.biases['decoder_b2']))
          return layer_2

     def run(self, x_data):
          encoder_op = self.encoder(self.X)
          decoder_op = self.decoder(encoder_op)

          y_pred = decoder_op
          y_true = self.X

          loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
          optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

          init = tf.global_variables_initializer()

          with tf.Session() as sess:
               sess.run(init)

               # train
               for i in range(1, self.num_steps+1):
                    for j in range(self.data_row - self.test_space):
                         sample, _ = sampling_arr(self.data, None, j, False)
                         sample = np.expand_dims(sample, axis=0) / self.dataMax

                         _, lo = sess.run([optimizer, loss], feed_dict = {self.X : sample})
                    print("  - step %i _ loss : %f" %(i, lo), end = "\r")

                    if( i % self.print_point == 0 ):
                         print("\n ===================== ")
               print("\n ===================== ")

               # test & compare
               y_decoder = []
               for j in range(self.data_row-self.test_space, self.data_row):
                    sample, _ = sampling_arr(self.data, None, j, False)
                    sample = np.expand_dims(sample, axis=0) / self.dataMax

                    y_AE = sess.run(decoder_op, feed_dict = {self.X : sample})
                    y_decoder.extend(y_AE)
               x_arr = x_data[ ((self.data_row-self.test_space) * self.dataAt.SRC_len)
                              : (self.data_row * self.dataAt.SRC_len) ]
               y_arr = self.data[ ((self.data_row-self.test_space) * self.dataAt.SRC_len)
                              : (self.data_row * self.dataAt.SRC_len) ]
          return x_arr, y_arr, list2numpy(y_decoder)
     
def autoEncoder_run(x_arr, srcData, test = 100, learning_rate = 0.01, step = 100, print_step = 10):
     AE = autoEncoder(srcData, test, learning_rate, step, print_step)
     AE.printOption()
     return AE.run(x_arr)
                    

















