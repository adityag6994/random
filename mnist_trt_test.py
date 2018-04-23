#test the formed engine on the image by running an infernece
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
from random import randint
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
from tensorrt import parsers
from tensorflow.python.framework import graph_util
import time

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
INPUT_H = 32
INPUT_W =  32
OUTPUT_SIZE = 10
DATA = '/home/d/Desktop/model_compare_caffe/svhn/svhn_test_images/'
IMAGE_MEAN = '/home/d/Desktop/model_compare_caffe/svhn/trt/svhn_trt.binaryproto'
RESULT_FILE = '/home/d/Desktop/model_compare_caffe/svhn/svhn_test_images.txt'
accuracy = 0

# def infer_caffe():


min_range = 0
max_range = 2005

f  = open(RESULT_FILE, "r")
fs = f.read()
words  = fs.split()
number = [int(w) for w in words]
print(len(number))

def inference_caffe(data):
	runtime = trt.infer.create_infer_runtime(G_LOGGER)
	engine = trt.utils.load_engine(G_LOGGER, "svhn_float32.engine")
	context = engine.create_execution_context()

	assert(engine.get_nb_bindings() == 2)
	img = data.astype(np.float32)
	output = np.empty(OUTPUT_SIZE, dtype = np.float32)

	d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
	# print(type(d_input))
	d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
	bindings = [int(d_input), int(d_output)]
	stream = cuda.Stream()
	cuda.memcpy_htod_async(d_input, img, stream)
	context.enqueue(1, bindings, stream.handle, None)
	cuda.memcpy_dtoh_async(output, d_output, stream)
	stream.synchronize()
	d_input.free()
	d_output.free()
	stream = None
	context.destroy()
	engine.destroy()
	runtime.destroy()
	return output	

# rand_file = randint(0,10)	
# path = DATA + str(rand_file) + '.png'
start = time.clock()
for current_image in range(min_range, max_range):
	# print(i)
	if(current_image%1001 == 1):
		print('currently processing image...', current_image, ' | accuracy : ', accuracy/current_image)#, ' | ', cuda.mem_get_info())

	path = DATA + str(current_image) + '.png'
	im = Image.open(path)
	# imshow(np.asarray(im))
	arr = np.array(im)
	img = arr.ravel()

	parser = parsers.caffeparser.create_caffe_parser()
	mean_blob = parser.parse_binary_proto(IMAGE_MEAN)  
	parser.destroy()
	#NOTE: This is different than the C++ API, you must provide the size of the data
	mean = mean_blob.get_data(INPUT_W ** 2)
	data = np.empty([INPUT_W ** 2])
	for i in range(INPUT_W ** 2):
	    data[i] = float(img[i]) - mean[i]
	mean_blob.destroy()

	output = inference_caffe(data)

	if(np.argmax(output) == number[current_image]):
		accuracy = accuracy+1;

stop = time.clock()

#print("runtime details : ", type(runtime))
# context.destroy()
# engine.destroy()
# runtime.destroy()

print('---------------------------------------------------------------------')
print('---------------------------------------------------------------------')
print('Accuracy :', accuracy/(max_range)*100, ' | Time : ', stop - start, 's')
print('---------------------------------------------------------------------')
