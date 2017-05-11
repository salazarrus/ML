from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	imageWidth = 28;
	imageHeigth = 28;
	inputVectorSize = imageWidth * imageHeigth;
	
	labelsSize = 10;
	# Create the model
	x = tf.placeholder(tf.float32, [None, inputVectorSize])
	W = tf.Variable(tf.zeros([inputVectorSize, labelsSize]))
	b = tf.Variable(tf.zeros([labelsSize]))
	y = tf.matmul(x, W) + b

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, labelsSize])

	learningRate = 0.5
	
	iterationsNumber = 1000;
	
	batchSize = 100;
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	
	descentStep = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

	sess = tf.InteractiveSession()
	
	tf.global_variables_initializer().run()
	
	# Traininig
	for _ in range(iterationsNumber):
		#take 100 examples of the test data, don't train on all 1000
		inputBatch, labelsBatch = mnist.train.next_batch(batchSize)
		
		sess.run(descentStep, feed_dict={x: inputBatch, y_: labelsBatch})

	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	
	# calculate mean of predictions
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	
	
	
#######################################	
############ THE PROGRAM ##############	
#######################################
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
	                help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)