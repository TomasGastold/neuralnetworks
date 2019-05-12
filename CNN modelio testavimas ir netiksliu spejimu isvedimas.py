import tensorflow as tf
import numpy as np
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def resizeimage(x):
    try:
        img = Image.open(x)
        plotis, aukštis = img.size

        img = img.resize((28, 28))
        print("Paveikslo", x, "dydis pakeistas į 28x28")
        img.save(x)
    except IOError:
        pass

def splitimage(x,y):
    try:
        im = Image.open(x)
        width, height = im.size
        numberOfSplits = y
        splitDist = width / numberOfSplits
        print(width, height)
        label = ""
        for i in range(0, numberOfSplits):
            x = splitDist * i
            y = 0
            w = splitDist + x
            h = height + y

            print(x, y, w, h)

            croppedImg = im.crop((x, y, w, h))  # Iškarpomas stačiakampis pagal x,y,w,h
            croppedImg.save("Skaitmuo" + str(i) + ".png")  # išsaugom
            resizeimage("Skaitmuo" + str(i) + ".png")

            data = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("Skaitmuo" + str(i) + ".png", flatten=True)))
            prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [data]})

            #Paverciam string formatu
            text = np.array2string(np.squeeze(prediction))
            #Konkatenuojam
            label = label + text

        print("Spėjimas:", label)

    except IOError:
        pass

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding ='SAME')
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides =[1,2,2,1], padding = 'SAME')

n_input = 784
n_output = 10

n_classes = 10

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
conv1 = tf.nn.leaky_relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)

conv2 = tf.nn.leaky_relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)

conv3 = tf.nn.leaky_relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
conv3 = maxpool2d(conv3)

fc = tf.reshape(conv3, [-1, 4*4*128])
fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

#Added name for using model later
output_layer = tf.matmul(fc, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.import_meta_graph('FINALCNN2.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nTestavimo rinkinio tikslumas:", test_accuracy)

#Testavimo rinkinio prognozės
y_test=(sess.run(output_layer,feed_dict={
                             X: mnist.test.images
                              }))

#Randami klaidingi spėjimai
idx=np.argmax(y_test,1)==np.argmax(mnist.test.labels,1)
cmp=np.where(idx==False)
# Išvedami klaidingi spėjimai
fig, axes = plt.subplots(5, 5, figsize=(20,20))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
cls_true=np.argmax(mnist.test.labels,1)[cmp]
cls_pred=np.argmax(y_test,1)[cmp]
images=mnist.test.images[cmp]
for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28,28), cmap='binary')
        xlabel = "Tikras: {0}, Spėjimas: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

#x -paveikslėlis, y- skaitmenys
splitimage("Pavekslėlis.png",1)
