import tensorflow.examples.tutorials.mnist.input_data
mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

#Set up Parameters
learningRate = 0.01
trainingLoop = 30
batchSize = 100

#Set Up Our Model
#Graph Input
image = tf.placeholder("float", [None, 784])#mnist input data is 28x28, therefore we use 748
digit = tf.placeholder("float", [None, 10]) #digits go from 0-9

#Set up weights
weights = tf.Variable(tf.zeros([784, 10])) #Weights represent the various probabilities of how the data flows
bias = tf.Variable(tf.zeros([10]))#Shifts regression line over time so it fits data


with tf.name_scope("logReg") as scope:
    #Liner model of our logistic regression
    model = tf.nn.softmax(tf.matmul(image, weights) + bias)

with tf.name_scope("Cost") as scope:
    #Use Cross Entrophy to Minimize Error Function
    cost = -tf.reduce_sum(digit*tf.log(model))

with tf.name_scope("Train") as scope:
    #Use Gradient Descent to Optimize Function
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

#Initialization
initialize = tf.initialize_all_variables()

#Training
with tf.Session() as Session:
    Session.run(initialize)

    #Cycle
    for loop in range(trainingLoop):
        print("training cycle " + str(loop))
        averageCost = 0.
        totalBatch = int(mnist.train.num_examples/batchSize)
        #Loop Through Batches
        for batch in range(totalBatch):
            batchImage, batchDigit = mnist.train.next_batch(batchSize)
            Session.run(optimizer, feed_dict={image: batchImage, digit: batchDigit})
            averageCost = Session.run(cost, feed_dict={image: batchImage, digit: batchDigit})/totalBatch

    print("TRAINING COMPLETED")

    #Test Model
    prediction = tf.equal(tf.argmax(model, 1), tf.argmax(digit, 1))

    #Calculate Accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print ("Accuracy", accuracy.eval({image: mnist.test.images, digit: mnist.test.labels}))