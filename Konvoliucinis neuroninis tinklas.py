import tensorflow as tf
import numpy as np
import scipy.ndimage
from PIL import Image

#Nuskaitom duomenų rinkinį
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded

# Strides parametras nurodo kaip langas judės
# Šiuo atvėju langas judės per 1 pikselį konvoliucinės funkcijos atvėju ir kas 2 pikselius surinkimo funkcijos atveju.
# Ksize yra surinkimo lango dydžio parametras. Šiuo atveju jis yra 2x2.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding ='SAME')
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides =[1,2,2,1], padding = 'SAME')

n_train = mnist.train.num_examples # 55,000
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000

n_input = 784   # įvesties sluoksnis (28x28 pikseliai)
n_output = 10   # išvesties sluoksnis (0-9 skaitmenys)

n_classes = 10

n_iterations = 100
batch_size = 128
dropout = 0.8

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
           'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
           'W_conv3': tf.Variable(tf.random_normal([5,5,64,128])),
           'W_fc': tf.Variable(tf.random_normal([4*4*128,1024])),
           'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
          'b_conv2': tf.Variable(tf.random_normal([64])),
          'b_conv3': tf.Variable(tf.random_normal([128])),
          'b_fc': tf.Variable(tf.random_normal([1024])),
          'out': tf.Variable(tf.random_normal([n_classes]))}

x = tf.reshape(X, shape=[-1, 28, 28, 1])

#Aktivacijos funkcija pridėta, tuomet neuronai nuspres iššauti ar ne ir siųsti išvestį į tolimesnį sluoksnį ar ne.
#Kiekvienas paslėptasis sluoksnis vykdys matricų daugybą tarp prieš tai buvusio sluoksnio išvesties ir einamojo sluoksnio svorių ir prides papildomą parametrą prie šių reikšmių.
conv1 = tf.nn.leaky_relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)

conv2 = tf.nn.leaky_relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)

conv3 = tf.nn.leaky_relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
conv3 = maxpool2d(conv3)

#Pagrindinis atmetimo funkcijos pliusas yra tas, kad ji sumažina neuroninio tinkle permokymo šansą. 
 #Atemetimo funkcijos parametras nuorodo slenkstį, kurį pasiekus mes atsitiktinai atmetam neuronus. 
# Pavyzdžiui naudojant 0.5 parametrą, kiekvienam neuronui ya 50% tikimybė būti atmestam, tačiau šiuo atvėju rezultatai buvo gauti geresni nenaudojant dropout funkcijos.
fc = tf.reshape(conv3, [-1, 4*4*128])
fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

output_layer = tf.matmul(fc, weights['out']) + biases['out']
# Lyginami spėjimai su tikromis reikšmėmis
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
#Default for Adamoptimizer learning rate is 0.001
#How quickly we want to lower the cost is determined by the learning rate.
# The lower the value for learning rate, the slower we will learn, and the more likely we'll get better results.
# Numatytas AdamOptimizer mokymosi greitis yra 0.001 
# Nuo mokymosi greičio priklauso kaip greitai mažės klaidų įvertis.
# Kuo mažesnis mokymosi greitis, tuo lėčiau neuroninis tinklas mokinsis ir tuo didesnė tikimybė gauti geresnius rezultatus.
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

#Argmax gražina maksimalų indeksą.
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
#tf.cast keičiamas kintamojo tipas
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

# treniruojama minimaliomis dalimis

for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # Atspausdinamas tikslumas ir klaidų įvertis minimaliose dalyse
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteracija", str(i), "\t| Klaidu ivertis =", str(minibatch_loss), "\t| Tikslumas =", str(minibatch_accuracy))
# Sukuriamas objektas, kuriame išsaugomi visi kintamieji
# Išsaugome modelį 
modelName = "FINALCNN2"
saver.save(sess, './'+modelName)

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nTestavimo rinkinio tikslumas:", test_accuracy)
