import tensorflow as tf
import numpy as np
import time, os

from plot import *
from numpy_process import list2numpy, arrArange
from run_sklearn import hgCluster_single_metric

project_ROOT = os.path.abspath(os.path.dirname(__file__))
learningLog = os.path.join(project_ROOT, "learning_log")

def printOption(learningName, learningRate, learningStep, data_row, damageList):
          print(" ========================== ")
          print(" " + str(learningName))
          print(" learning rate : " + str(learningRate))
          print(" train step : " + str(learningStep))
          print(" data row length : " + str(data_row))
          print(" damaged item id : " + str(damageList))
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
          np.random.shuffle(self.data)
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
               "encoder_h1" : tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1], stddev=0.01 )),
               "encoder_h2" : tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2], stddev=0.01)),
               "decoder_h1" : tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1], stddev=0.01)),
               "decoder_h2" : tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input], stddev=0.01))
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

     def run(self, test = False, save = False, load = False):
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

               if load:
                    ckpt = tf.train.get_checkpoint_state(os.path.join("learning_log", "autoEncoder"))
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
                         if save:
                              save(self.LOG_DIR, saver, sess, i)
               print("\n ===================== ")

               # encoder, decoder
               y_encoder = []
               y_decoder = []
               for j in range(self.data_row):
                    sample = np.expand_dims(self.data[j], axis=0) / self.dataMax

                    y_en, y_de = sess.run([encoder_op, decoder_op], feed_dict = {self.X : sample})
                    y_encoder.extend(y_en)
                    y_decoder.extend(y_de)

               # compare test set decoder
               if(test == True):
                    test_decoder = []
                    for j in range(int(self.data_row - self.testSpace), self.data_row):
                         sample = np.expand_dims(self.data[j], axis=0) / self.dataMax

                         test_AE = sess.run(decoder_op, feed_dict = {self.X : sample})
                         test_decoder.extend(test_AE)

                    de_arr = list2numpy(test_decoder)
                    fig, plot = plt.subplots(1,2)
                    for i in range(self.data_row-self.testSpace):
                         line(plot[0], np.arange(self.num_input), self.data[i], None, "training", None, 1)
                    for i in range(self.testSpace):
                         line(plot[1], np.arange(self.num_input), de_arr[i], None, "test", None, 1)
                    fig.canvas.set_window_title("compare test")
                    plt.savefig(self.LOG_DIR + "/sample.png", bbox_inches="tight")
                    
          newY_en = list2numpy(y_encoder) * self.dataMax
          newY_de = list2numpy(y_decoder) * self.dataMax
          return newY_en, newY_de, self.num_hidden_2, self.num_steps
     
def autoEncoder_run(src_arr, itemId, test, learning_rate, step, print_step, damageList):
     AE = autoEncoder(src_arr, itemId, test, learning_rate, step, print_step, damageList)
     printOption("auto-encoder", learning_rate, step, src_arr.shape[0], damageList)
     encoded_data, decoded_data, n_encode, step = AE.run()

     fig, plot = create_fig(1,3)
     legend_list = []
     if ( damageList == []):
          hgCluster_labels = hgCluster_single_metric(src_arr, 3)
          for label, color, label_name in zip(arrArange(3), "byg", ["label 1", "label 2", "label 3"]):
               legend_list.append(legend_label(color, label_name))
     
               origin = src_arr[hgCluster_labels == label].T
               encoded = encoded_data[hgCluster_labels == label].T
               decoded = decoded_data[hgCluster_labels == label].T
               
               line(plot[0], origin, None, label, "original", color, 1, option="singleArr")
               line(plot[1], encoded, None, label, "encoded", color, 1, option = "singleArr")
               line(plot[2], decoded, None, label, "dncoded", color, 1, option = "singleArr")
     else:
          for i in range(src_arr.shape[0]):
               if( str(itemId[i]) in damageList):
                    line(plot[0], arrArange(src_arr.shape[1]), src_arr[i], itemId[i], "original", "red", 2)
                    line(plot[1], arrArange(n_encode), encoded_data[i], itemId[i], "encoded", "red", 2)
                    line(plot[2], arrArange(n_encode), dncoded_data[i], itemId[i], "decoded", "red", 2)
               else:
                    line(plot[0], arrArange(src_arr.shape[1]), src_arr[i], itemId[i], "original", "blue", 0.1)
                    line(plot[1], arrArange(n_encode), encoded_data[i], itemId[i], "encoded", "blue", 0.1)
                    line(plot[2], arrArange(n_encode), decoded_data[i], itemId[i], "decoded", "blue", 0.1)
               print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(src_arr.shape[0]) , end = "\r")
          print("\n")
          
          for color, label_name in zip(["blue", "red"], ["normal", "damage"]):
               legend_list.append(legend_label(color, label_name))
          
                                  
     plot_legend(plot[0], legend_list)
     plot_legend(plot[1], legend_list)
     plot_legend(plot[2], legend_list)
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
          np.random.shuffle(self.data)
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
          
          self.num_hidden1 = 128
          self.num_hidden2 = 128
          self.num_input = int(self.data.shape[1])
          self.num_noise = 64

          self.X = tf.placeholder(tf.float32, [1, self.num_input])
          self.Z = tf.placeholder(tf.float32, [1, self.num_noise])

          self.G = {
               "W1" : tf.get_variable('g_w1', [self.num_noise, self.num_hidden1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
               "b1" : tf.get_variable('g_b1', [self.num_hidden1], initializer=tf.zeros_initializer()),
               "W2" : tf.get_variable('g_w2', [self.num_hidden1, self.num_hidden2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
               "b2" : tf.get_variable('g_b2', [self.num_hidden2], initializer=tf.zeros_initializer()),
               "W3" : tf.get_variable('g_w3', [self.num_hidden2, self.num_input], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
               "b3" : tf.get_variable('g_b3', [self.num_input], initializer=tf.zeros_initializer())
               }
          self.D = {
               "W1" : tf.get_variable('d_w1', [self.num_input, self.num_hidden1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
               "b1" : tf.get_variable('d_b1', [self.num_hidden1], initializer=tf.zeros_initializer()),
               "W2" : tf.get_variable('d_w2', [self.num_hidden1, self.num_hidden2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
               "b2" : tf.get_variable('d_b2', [self.num_hidden2], initializer=tf.zeros_initializer()),
               "W3" : tf.get_variable('d_w3', [self.num_hidden2, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)),
               "b3" : tf.get_variable('d_b3', [1], initializer=tf.zeros_initializer())
               }

          self.LOG_DIR = os.path.join(learningLog, "GAN")

     def generator(self, noise):
          hidden1 = tf.nn.relu(
               tf.matmul(noise, self.G['W1']) + self.G['b1'])
          hidden2 = tf.nn.relu(
               tf.matmul(hidden1, self.G['W2']) + self.G['b2'])
          output = tf.matmul(hidden2, self.G['W3']) + self.G['b3']
          return output

     def discriminator(self, inputs):
          hidden1 = tf.nn.relu(
               tf.matmul(inputs, self.D['W1']) + self.D['b1'])
          hidden2 = tf.nn.relu(
               tf.matmul(hidden1, self.D['W2']) + self.D['b2'])
          output = tf.matmul(hidden2, self.D['W3']) + self.D['b3']
          return output

     def create_noise(self, mean, scale):
          return np.random.normal(mean, scale, size = (1, self.num_noise))

     def run(self, test = True, save = False, load = False):
          G = self.generator(self.Z)
          D_real = self.discriminator(self.X)
          D_fake = self.discriminator(G)
          
          G_var_list = [self.G['W1'], self.G['b1'], self.G['W2'], self.G['b2'], self.G['W3'], self.G['b3']]
          D_var_list = [self.D['W1'], self.D['b1'], self.D['W2'], self.D['b2'], self.D['W3'], self.D['b3']]

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
               tf.summary.scalar("G_loss", loss_G)
               tf.summary.scalar("D_loss_real", loss_D_real)
               tf.summary.scalar("D_loss_fake", loss_D_fake)

               summary_op = tf.summary.merge_all()
               summary_writer = tf.summary.FileWriter(os.path.join("learning_log", "GAN"))
               sess.run(init)
               if load:
                    ckpt = tf.train.get_checkpoint_state(os.path.join("learning_log", "GAN"))
                    if ckpt and ckpt.model_checkpoint_path:
                         saver.restore(sess, ckpt.model_checkpoint_path)
                         print(" -> Model restored...")

               for i in range(1, self.num_steps+1):
                    step_start = time.time()
                    noise = self.create_noise(0.5, 0.01)
                    for j in range(int(self.data_row)):
                         sample = np.expand_dims(self.data[j], axis=0) / self.dataMax
                         
                         _, __, cost_D, loss_D = sess.run([train_D_real, train_D_fake, loss_D_real, loss_D_fake],
                                                  feed_dict={self.X : sample, self.Z : noise})
                         _ = sess.run(train_G, feed_dict={self.Z : noise})

                    print("  - step %i _ real-cost : %f fake-loss : %f" %(i, cost_D, loss_D), end = "\r")

                    if( i % self.print_point == 0 ):
                         point_step_final = time.time()
                         print("\n time : " + str( (point_step_final - step_start) * self.print_point) )
                         print(" ===================== ")
                         if save:
                              save(self.LOG_DIR, saver, sess, i)

                         if( test == True):
                              noise = self.create_noise(0.5, 0.01)
                              fake_waveForm = sess.run(G, feed_dict={self.Z : noise})
                              fig, plot = create_fig(None, None)
                              line(plot, arrArange(self.num_input), (fake_waveForm[0] * self.dataMax), "GAN test",
                                   "fake wave form" , None, 1)
                              sampleImg_root = os.path.join(self.LOG_DIR, "img")
                              if not os.path.exists(sampleImg_root):
                                   os.mkdir(sampleImg_root)
                              plt.savefig(sampleImg_root + "/sample_{}.png".format(i), bbox_inches="tight")
                              plt.close()
                              print(" \n create fake wave-form to project/learning_log/GAN/sample_{}.png".format(i)  )
               print("\n ===================== ")

               fake_arr = np.zeros([self.data_row, self.data.shape[1]])
               for i in range(self.data_row):
                    sample = np.expand_dims(self.data[i], axis=0) / self.dataMax
                    noise = self.create_noise(0.5, 0.01)
                    fake_arr[i] = sess.run(G, feed_dict={self.Z : noise})
               return fake_arr * self.dataMax

def GAN_run(src_arr, itemId, test, learning_rate, step, print_step, damageList):
     GAN = GenerativeAdversarialNet(src_arr, itemId, test, learning_rate, step, print_step, damageList)
     printOption("Generative Adversarial Network", learning_rate, step, src_arr.shape[0], damageList)
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

# Deep Neural Network
class DeepNeuralNet:
     def __init__(self, data, itemId, test = 100, learning_rate = 0.001, step = 100,
                  print_step = 10):
          # item id
          self.itemId = itemId

          # src data
          self.data, self.labels = self.expandDamageWaveForm(
               data, hgCluster_single_metric(data, 3), Damage_label = 1, multiple = 100)
          self.data_row = int(self.data.shape[0])
          self.dataMax = self.data.max()
          self.testSpace = test

          self.one_hot_labels = np.zeros([self.data_row, 3])
          for i in range(len(self.labels)):
               for j in range(3):
                    if (self.labels[i] == j):
                         self.one_hot_labels[i, j] = 1
                         break

          # run test
          if ( self.data_row < test ):
               print("\n error : test row must small then total row \n")
               return None
          
          self.learning_rate = learning_rate
          self.num_steps = step
          self.print_point = print_step

          self.num_input = int(self.data.shape[1])
          self.num_hidden1 = 128
          self.num_hidden2 = 256
          self.num_hidden3 = 512
          self.num_class = 3

          self.X = tf.placeholder(tf.float32, [1, self.num_input])
          self.y = tf.placeholder(tf.float32, [1, 3])

          self.W1 = tf.get_variable('w1', [self.num_input, self.num_hidden1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
          self.b1 = tf.get_variable('b1', [self.num_hidden1], initializer=tf.zeros_initializer())
          self.W2 = tf.get_variable('w2', [self.num_hidden1, self.num_hidden2], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
          self.b2 = tf.get_variable('b2', [self.num_hidden2], initializer=tf.zeros_initializer())
          self.W3 = tf.get_variable('w3', [self.num_hidden2, self.num_hidden3], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
          self.b3 = tf.get_variable('b3', [self.num_hidden3], initializer=tf.zeros_initializer())
          self.W4 = tf.get_variable('w4', [self.num_hidden3, self.num_class], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
          self.b4 = tf.get_variable('b4', [self.num_class], initializer=tf.zeros_initializer())

          self.layer1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
          self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
          self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.W3) + self.b3)
          self.fc = tf.matmul(self.layer3, self.W4) + self.b4
          self.softmax = tf.nn.softmax(self.fc)

          self.LOG_DIR = os.path.join(learningLog, "DNN")

     def expandDamageWaveForm(self, data, labels, Damage_label, multiple):
          tempList = []
          for i in range(len(labels)):
               if ( labels[i] == Damage_label):
                    for j in range(multiple):
                         mask = np.random.normal(0, 0.001, size = (data.shape[1]))
                         temp = data[i] + mask
                         tempList.append(temp)
          tempArr = list2numpy(tempList)
          newLabel = np.zeros([ data.shape[0] + tempArr.shape[0] ])
          newLabel[:data.shape[0]] = labels
          newLabel[data.shape[0]:] = np.full_like(np.arange(tempArr.shape[0], dtype=int), Damage_label)
          newData = np.zeros([ data.shape[0] + tempArr.shape[0], data.shape[1] ])
          newData[:data.shape[0]] = data
          newData[data.shape[0]:] = tempArr

          shuffleIndex = np.random.permutation(len(newLabel))
          return newData[shuffleIndex], newLabel[shuffleIndex]

     def getNewData(self):
          return self.data, self.labels

     def run(self, test = True, save = False, load = False):
          varList = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
               logits = self.fc, labels = self.y))
          train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
               loss, var_list = varList)
          sess = tf.Session()
          sess.run(tf.global_variables_initializer())

          if load:
               ckpt = tf.train.get_checkpoint_state(os.path.join("learning_log", "DNN"))
               if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(" -> Model restored...")

          loss_list = []
          for i in range(1, self.num_steps):
               step_start = time.time()
               for j in range(self.data_row-self.testSpace):
                    sample = np.expand_dims(self.data[j], axis=0) / self.dataMax
                    sample_label = np.expand_dims(self.one_hot_labels[j], axis=0)

                    train_loss, _ = sess.run([loss, train_op],
                                             feed_dict = {self.X : sample, self.y : sample_label})
               loss_list.append(train_loss)

               print("  - step %i _ loss : %f" %(i, train_loss), end = "\r")

               if( i % self.print_point == 0 ):
                    point_step_final = time.time()
                    print("\n time : " + str( (point_step_final - step_start) * self.print_point) )
                    print(" ===================== ")
                    if save:
                         save(self.LOG_DIR, saver, sess, i)

          if( test == True):
               test_result_list = []
               test_start = time.time()
               for j in range(self.data_row-self.testSpace, self.data_row):
                    sample = np.expand_dims(self.data[j], axis = 0) / self.dataMax
                    sample_label = np.expand_dims(self.one_hot_labels[j], axis = 0)

                    test_loss, test_result = sess.run([loss, self.softmax], feed_dict = {self.X : sample, self.y : sample_label})
                    test_result_list.extend(test_result)

               test_labels = tf.placeholder(tf.float32, [None, 3])
               predict = list2numpy(test_result_list)
               correct = tf.equal(tf.argmax(test_labels, 1), tf.argmax(test_labels, 1))
               acc = tf.reduce_mean(tf.cast(correct, tf.float32))
               accuracy = sess.run(acc, feed_dict = {test_labels : predict,
                                                     test_labels : self.one_hot_labels[self.data_row-self.testSpace:]})

               test_final = time.time()
               print("\n ===================== ")
               print(" TEST set Loss : %f ,  Accuracy : %f" %(test_loss, accuracy))
               print(" time : " + str( test_final - test_start))
               
          return self.data, list2numpy(loss_list), predict 

def DNN_run(src_arr, itemId, test, learning_rate, step, print_step, damageList):
     DNN = DeepNeuralNet(src_arr, itemId, test, learning_rate, step, print_step)
     printOption("Deep Neural Network", learning_rate, step, DNN.data.shape[0], damageList)
     data, loss, predict = DNN.run()
     
     fig, plot = create_fig(1,2)
     line(plot[0], loss, None, "loss", "DNN-loss", None, 1, option="singleArr")
     for i in range(predict.shape[0]):
          if( np.argmax(predict[i]) == 1):
               setColor = "red"
          elif( np.argmax(predict[i]) == 2):
               setColor = "green"
          else:
               setColor = "blue"
          line(plot[1], np.arange(data.shape[1]), data[data.shape[0]-predict.shape[0]+i], "predict", "DNN-result", setColor, 1)
          print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(predict.shape[0]) , end = "\r")
     print("\n")
     return fig, step
     













