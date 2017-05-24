import tensorflow as tf
from readData import *


images2 = readImages2('train-images.idx3-ubyte')
labels = readLabels('train-labels.idx1-ubyte')
labels = oneHot(labels)

#images2 = np.c_[np.ones([10000,1]),images2]

print('Read Data Done.')
"""
def addLayer(inputData,inSize,outSize,activity_function = None):  
    Weights = tf.Variable(tf.random_normal([inSize,outSize]),'float32')   
    basis = tf.Variable(tf.zeros([1,outSize])+0.1,'float32')    
    weights_plus_b = tf.matmul(inputData,Weights)+basis  
    if activity_function is None:  
        ans = weights_plus_b  
    else:  
        ans = activity_function(weights_plus_b)  
    return ans  

x = tf.placeholder('float32',[None,28*28])
y = tf.placeholder('float32',[None,10])

print('init x,y')

hideL1 = addLayer(x,28*28,100,tf.nn.relu)
hideL2 = addLayer(hideL1,100,10,tf.nn.relu)
loss = tf.reduce_mean(tf.reduce_sum(tf.square((y-hideL2)),reduction_indices = [1]))  
train =  tf.train.GradientDescentOptimizer(0.1).minimize(loss) 

print('init train')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10):
    trainX,trainY = getRandomData(images2,labels,100)
    sess.run(train,feed_dict={x:trainX,y:trainY})
    print(sess.run(loss,feed_dict={x:images2,y:labels}))

"""
w1 = tf.Variable(tf.random_normal([28*28,100],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([100,10],stddev = 1,seed = 1))



x = tf.placeholder(tf.float32,shape=(None,28*28),name='x_input')
y = tf.placeholder(tf.float32,shape=(None,10),name = 'y_input')

basis1 = tf.Variable(tf.zeros([100],'float32'))
a = tf.sigmoid(tf.matmul(x,w1)+basis1)

#a = tf.concat(1, [tf.ones([None,1]), a])
basis2 = tf.Variable(tf.zeros([10],'float32'))
y_ = tf.nn.softmax(tf.matmul(a,w2)+basis2)

cross_entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_,1e-10,1.0)))
tf_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#tf_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = getRandomData(images2,labels,200)
    sess.run(tf_step,feed_dict={x:batch_xs,y:batch_ys})
    if i%100==0:
        step_cross_entropy = sess.run(cross_entropy,feed_dict={x:images2,y:labels})
        print ('%d : %f'%(i,step_cross_entropy))
        
step_cross_entropy = sess.run(cross_entropy,feed_dict={x:images2,y:labels})
print ('%d : %f'%(i,step_cross_entropy))

resultW1 = sess.run(w1)
resultW2 = sess.run(w2)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('success:')
print(sess.run(accuracy, feed_dict={x: images2, y: labels}))

testImages = readImages2('t10k-images.idx3-ubyte')
testlabels = readLabels('t10k-labels.idx1-ubyte')
testlabels = oneHot(testlabels)
print('test Data Tests:')
print(sess.run(accuracy, feed_dict={x: testImages, y: testlabels}))