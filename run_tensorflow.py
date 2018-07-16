import tensorflow as tf
import numpy as np
import time, os

from plot import *
from numpy_process import list2numpy

project_ROOT = os.path.abspath(os.path.dirname(__file__))
learningLog = os.path.join(project_ROOT, "learning_log")

def printOption(learningName, learningRate, learningStep, data_row):
          print(" ========================== ")
          print(" " + str(learningName))
          print(" learning rate : " + str(learningRate))
          print(" train step : " + str(learningStep))
          print(" data row length : " + str(data_row))
          print(" ========================== ")

def save(log_dir, saver, sess, step):
     if not (os.path.exists(log_dir)):
          os.mkdir(log_dir)
     saver.save(sess, log_dir + "/model.ckpt", step)
     #print(" \n saved " + str(log_dir + "/model.ckpt" + step))
          
class autoEncoder:
     def __init__(self, data, itemId, test = 100, learning_rate = 0.01, step = 1000,
                  print_step = 10, damageList = []):
          # item id
          self.itemId = itemId
          self.damageList = damageList
          
          # src data
          self.data = data
          self.data_row = int(self.data.shape[0])
          self.dataMax = self.data.max()
          self.testSpace = test

          # run test
          if ( self.data_row < test ):
               print(" test row must small then total row ")
               return None
          
          self.learning_rate = learning_rate
          self.num_steps = step
          self.print_point = print_step
          
          self.num_hidden_1 = 256
          self.num_hidden_2 = 128
          self.num_input = int(self.data.shape[1])

          self.X = tf.placeholder("float", [1, self.num_input])

          self.weights = {
               "encoder_h1" : tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
               "encoder_h2" : tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
               "decoder_h1" : tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
               "decoder_h2" : tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input]))
               }
          self.biases = {
               "encoder_b1" : tf.Variable(tf.zeros([self.num_hidden_1])),
               "encoder_b2" : tf.Variable(tf.zeros([self.num_hidden_2])),
               "decoder_b1" : tf.Variable(tf.zeros([self.num_hidden_1])),
               "decoder_b2" : tf.Variable(tf.zeros([self.num_input]))
               }

          self.LOG_DIR = os.path.join(learningLog, "autoEncoder")
     
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

     def run(self, test = True):
          encoder_op = self.encoder(self.X)
          decoder_op = self.decoder(encoder_op)

          y_pred = decoder_op
          y_true = self.X

          loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
          optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

          init = tf.global_variables_initializer()
          saver = tf.train.Saver()
          with tf.Session() as sess:
               sess.run(init)
               ckpt = tf.train.get_checkpoint_state(self.LOG_DIR)
               if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Model restored...")

               # train
               for i in range(1, self.num_steps+1):
                    step_start = time.time()
                    for j in range(int(self.data_row - self.testSpace)):
                         sample = np.expand_dims(self.data[j], axis=0) / self.dataMax

                         _, lo = sess.run([optimizer, loss], feed_dict = {self.X : sample})
                    print("  - step %i _ loss : %f" %(i, lo), end = "\r")
                    

                    if( i % self.print_point == 0 ):
                         point_step_final = time.time()
                         print("\n time : " + str( (point_step_final - step_start) * self.print_point) )
                         print(" ===================== ")
                         save(self.LOG_DIR, saver, sess, i)
               print("\n ===================== ")

               # encoder
               y_encoder = []
               for j in range(self.data_row):
                    sample = np.expand_dims(self.data[j], axis=0) / self.dataMax

                    y_AE = sess.run(encoder_op, feed_dict = {self.X : sample})
                    y_encoder.extend(y_AE)

               # compare decoder
               if(test == True):
                    y_decoder = []
                    for j in range(int(self.data_row - self.testSpace), self.data_row):
                         sample = np.expand_dims(self.data[j], axis=0) / self.dataMax

                         y_AE = sess.run(decoder_op, feed_dict = {self.X : sample})
                         y_decoder.extend(y_AE)

                    de_arr = list2numpy(y_decoder)
                    fig, plot = plt.subplots(1,2)
                    for i in range(self.data_row-self.testSpace):
                         plot[0].plot( np.arange(self.num_input), self.data[i])
                    for i in range(self.testSpace):
                         plot[1].plot( np.arange(self.num_input), de_arr[i] )
                    fig.canvas.set_window_title("compare test")
                    
          newY_arr = list2numpy(y_encoder) * self.dataMax    
          return newY_arr, self.num_hidden_2, self.num_steps
     
def autoEncoder_run(src_arr, itemId, test, learning_rate, step, print_step, damageList):
     AE = autoEncoder(src_arr, itemId, test, learning_rate, step, print_step, damageList)
     printOption("auto-encoder", learning_rate, step, src_arr.shape[0])
     encoded_data, n_encode, step = AE.run()
     
     compareName = "original"
     tick_arr = arrArange(src_arr.shape[1])
     fig, plot = create_fig(1,2)
     for i in range(src_arr.shape[0]):
          if( str(itemId[i]) in damageList):
               line(plot[0], tick_arr, src_arr[i], itemId[i], compareName, "red", 2)
               line(plot[1], arrArange(n_encode), encoded_data[i], itemId[i], "encoded", "red", 2)
          else:
               line(plot[0], tick_arr, src_arr[i], itemId[i], compareName, "blue", 0.1)
               line(plot[1], arrArange(n_encode), encoded_data[i], itemId[i], "encoded", "blue", 0.1)
          print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(src_arr.shape[0]) , end = "\r")
     print("\n")
     return fig, step

#"Generative Adversarial Networks."
# Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu,
# David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014
class GenerativeAdversarialNet:
     def __init__(self, data, itemId, test = 100, learning_rate = 0.0002, step = 1000,
                  print_step = 10, damageList = []):
          # item id
          self.itemId = itemId
          self.damageList = damageList
          
          # src data
          self.data = data
          self.data_row = int(self.data.shape[0])
          self.dataMax = self.data.max()
          self.testSpace = test
          
          # run test
          if ( self.data_row < test ):
               print(" test row must small then total row ")
               return None
          
          self.learning_rate = learning_rate
          self.num_steps = step
          self.print_point = print_step
          
          self.num_hidden = 256
          self.num_input = int(self.data.shape[1])
          self.num_noise = 128

          self.X = tf.placeholder(tf.float32, [1, self.num_input])
          self.Z = tf.placeholder(tf.float32, [1, self.num_noise])

          self.G = {
               "W1" : tf.Variable(tf.random_normal([self.num_noise, self.num_hidden], stddev=0.01)),
               "b1" : tf.Variable(tf.zeros([self.num_hidden])),
               "W2" : tf.Variable(tf.random_normal([self.num_hidden, self.num_input], stddev=0.01)),
               "b2" : tf.Variable(tf.zeros(self.num_input))
               }
          self.D = {
               "W1" : tf.Variable(tf.random_normal([self.num_input, self.num_hidden], stddev=0.01)),
               "b1" : tf.Variable(tf.zeros([self.num_hidden])),
               "W2" : tf.Variable(tf.random_normal([self.num_hidden, 1], stddev=0.01)),
               "b2" : tf.Variable(tf.zeros([1]))
               }

          self.LOG_DIR = os.path.join(learningLog, "GAN")

     def generator(self, noise):
          hidden = tf.nn.relu(
               tf.matmul(noise, self.G['W1']) + self.G['b1'])
          output = tf.nn.sigmoid(
               tf.matmul(hidden, self.G['W2']) + self.G['b2'])
          return output

     def discriminator(self, inputs):
          hidden = tf.nn.relu(
               tf.matmul(inputs, self.D['W1']) + self.D['b1'])
          output = tf.nn.sigmoid(
               tf.matmul(hidden, self.D['W2']) + self.D['b2'])
          return output

     def create_noise(self):
          return np.random.normal(size = (1, self.num_noise))

     def run(self, test = True):
          G = self.generator(self.Z)
          D_real = self.discriminator(self.X)
          D_fake = self.discriminator(G)
          
          G_var_list = [self.G['W1'], self.G['b1'], self.G['W2'], self.G['b2']]
          D_var_list = [self.D['W1'], self.D['b1'], self.D['W2'], self.D['b2']]

          loss_D_real = tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real) ))
          loss_D_fake = tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.zeros_like(D_fake) ))
          loss_G = tf.reduce_mean(
               tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.ones_like(D_fake) ))

          train_D_real = tf.train.AdamOptimizer(self.learning_rate).minimize(
               loss_D_real, var_list = D_var_list)
          train_D_fake = tf.train.AdamOptimizer(self.learning_rate).minimize(
               loss_D_fake, var_list = D_var_list)
          train_G = tf.train.AdamOptimizer(self.learning_rate).minimize(
               loss_G, var_list = G_var_list)

          init = tf.global_variables_initializer()
          saver = tf.train.Saver()
          with tf.Session() as sess:
               sess.run(init)

               loss_var_D, loss_var_G = 0, 0
               ckpt = tf.train.get_checkpoint_state(self.LOG_DIR)
               if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(" -> Model restored...")

               for i in range(1, self.num_steps+1):
                    step_start = time.time()
                    for j in range(int(self.data_row)):
                         sample = np.expand_dims(self.data[j], axis=0) / self.dataMax
                         noise = self.create_noise()

                         _, __, loss_real, loss_fake = sess.run([train_D_real, train_D_fake, loss_D_real, loss_D_fake],
                                                  feed_dict={self.X : sample, self.Z : noise})
                         _, loss_G_real = sess.run([train_G, loss_G], feed_dict={self.Z : noise})

                    print("  - step %i _ D-loss : %f G-loss : %f" %(i, loss_real, loss_G_real), end = "\r")

                    if( i % self.print_point == 0 ):
                         point_step_final = time.time()
                         print("\n time : " + str( (point_step_final - step_start) * self.print_point) )
                         print(" ===================== ")
                         save(self.LOG_DIR, saver, sess, i)

                         if( test == True):
                              noise = self.create_noise()
                              fake_waveForm = sess.run(G, feed_dict={self.Z : noise}) #* self.dataMax
                              fig, plot = create_fig(None, None)
                              line(plot, arrArange(self.num_input), fake_waveForm[0], "GAN test",
                                   "fake wave form" , None, 1)
                              sampleImg_root = os.path.join(self.LOG_DIR, "img")
                              if not os.path.exists(sampleImg_root):
                                   os.mkdir(sampleImg_root)
                              plt.savefig(sampleImg_root + "/sample_{}.png".format(i), bbox_inches="tight")
                              print(" \n create fake wave-form to project/learning_log/GAN/sample_{}.png".format(i)  )
               print("\n ===================== ")

               fake_arr = np.zeros([self.data_row, self.data.shape[1]])
               for i in range(self_data_row):
                    sample = np.expand_dims(self.data[i], axis=0) / self.dataMax
                    noise = self.create_noise()
                    fake_arr[i] = sess.run(G, feed_dict={self.Z : noise})
               return fake_arr * self.dataMax

def GAN_run(src_arr, itemId, test, learning_rate, step, print_step, damageList):
     GAN = GenerativeAdversarialNet(src_arr, itemId, test, learning_rate, step, print_step, damageList)
     printOption("Generative Adversarial Network", learning_rate, step, src_arr.shape[0])
     fake_arr = GAN.run()

     compareName = "original"
     tick_arr = arrArange(src_arr.shape[1])
     fig, plot = create_fig(1,2)
     for i in range(src_arr.shape[0]):
          if( str(itemId[i]) in damageList):
               line(plot[0], tick_arr, src_arr[i], itemId[i], compareName, "red", 2)
               line(plot[1], tick_arr, fake_arr[i], itemId[i], "fake", "red", 2)
          else:
               line(plot[0], tick_arr, src_arr[i], itemId[i], compareName, "blue", 0.1)
               line(plot[1], tick_arr, fake_arr[i], itemId[i], "fake", "blue", 0.1)
          print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(src_arr.shape[0]) , end = "\r")
     print("\n")
     return fig, step
















