"""
	MNIST Test using trained Caffe model. 
	We are intrested in bacth size of one and
	the over all accuracy.
	Date : 16 April 2018
"""
import lmdb
import time
import math
import caffe
import timeit
import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
import matplotlib.patches as mpatches
from google.protobuf import text_format

#Initialise Caffe using GPU
caffe.set_device(0)
caffe.set_mode_gpu()
caffe.set_random_seed(0)
np.random.seed(0)
print('Initialized Caffe!')

model = 'mnist_model.prototxt'
weights = 'mnist_model.caffemodel'
RESULT_FILE = 'mnist_test_images.txt'

f  = open(RESULT_FILE, "r")
fs = f.read()
words  = fs.split()
number = [int(w) for w in words]

net = caffe.Net(model, weights, caffe.TEST)

min_value = 0
max_value = 10000
accuracy = 0

start = time.clock()
for current_image in range(min_value, max_value):
	if(current_image%1001 == 1):
		print('currently processing image...', current_image, ' | accuracy : ', accuracy/current_image)#, ' | ', cuda.mem_get_info())

	# im = np.array(Image.open('data/0.png'))
	im = np.array(Image.open( 'data/' + str(current_image) + '.png'))
	im_input = im[np.newaxis, np.newaxis, :, :]

	net.blobs['data'].reshape(*im_input.shape)
	net.blobs['data'].data[...] = im_input

	output = net.forward()
	# print(output['loss'].argmax(), ' ', number[current_image])

	if(output['loss'].argmax() == number[current_image]):
		accuracy = accuracy+1;

stop = time.clock()
print ('accuracy : ', accuracy/(max_value - min_value), ' | Total Time : ', stop-start, ' | Time for Inference: ', (stop-start)/(max_value-min_value))