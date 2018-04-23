##############################################################################
##### Create engine from .caffemodel and .prototxt file, and save it #########
##############################################################################
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
from random import randint
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
from tensorrt import parsers
import sys
import time

# Initilisations

# Logging Initilisation : To keep track of logs
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

# Size Initilisation : Input Output Layers
INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['loss']
#Size of image
INPUT_H = 28
INPUT_W =  28
#Number of outputs
OUTPUT_SIZE = 10

# File Path : Path to prototxt and weight files
MODEL_PROTOTXT = 'mnist_model.prototxt'
CAFFE_MODEL = 'mnist_model.caffemodel'
DATA = 'data/'
IMAGE_MEAN = 'mnist_model.binaryproto'
RESULT_FILE = 'mnist_test_images.txt'

f  = open(RESULT_FILE, "r")
fs = f.read()
words  = fs.split()
number = [int(w) for w in words]

# Creating Engine : By parsing .caffemodel and .prototxt
engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
                                       MODEL_PROTOTXT,
                                       CAFFE_MODEL,
                                       1,
                                       1 << 20,
                                       OUTPUT_LAYERS,
                                       trt.infer.DataType.INT8)

min_range = 0
max_range = 10000
accuracy  = 0
start = time.clock()

for current_image in range(min_range, max_range):
	# print(i)
	if(current_image%501 == 1):
		print('currently processing image...', current_image, ' | accuracy : ', accuracy/current_image)#, ' | ', cuda.mem_get_info())

	path = DATA + str(current_image) + '.png'
	im = Image.open(path)
	# imshow(np.asarray(im))
	arr = np.array(im)
	img = arr.ravel()

	# parser = parsers.caffeparser.create_caffe_parser()
	# mean_blob = parser.parse_binary_proto(IMAGE_MEAN)  
	# parser.destroy()
	# #NOTE: This is different than the C++ API, you must provide the size of the data
	# mean = mean_blob.get_data(INPUT_W ** 2)
	# data = np.empty([INPUT_W ** 2])
	# for i in range(INPUT_W ** 2):
	#     data[i] = float(img[i]) - mean[i]
	# mean_blob.destroy()

	# Run inference
	runtime = trt.infer.create_infer_runtime(G_LOGGER)
	context = engine.create_execution_context()
	assert(engine.get_nb_bindings() == 2)
	#convert input data to Float32
	img = img.astype(np.float32)
	#create output array to receive data
	output = np.empty(OUTPUT_SIZE, dtype = np.float32)

	# Get memory from GPU
	d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
	d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
	bindings = [int(d_input), int(d_output)]
	stream = cuda.Stream()

	# Transfer data to that memory in GPU
	#transfer input data to device
	cuda.memcpy_htod_async(d_input, img, stream)
	#execute model
	context.enqueue(1, bindings, stream.handle, None)
	#transfer predictions back
	cuda.memcpy_dtoh_async(output, d_output, stream)
	#syncronize threads
	stream.synchronize()
	
	if(np.argmax(output) == number[current_image]):
		accuracy = accuracy+1;
	
	# print ("Prediction: " + str(np.argmax(output)), ' ', number[current_image],  ' ', path)

	context.destroy()
	runtime.destroy()
stop = time.clock()

print(accuracy)
print(max_range)
print('final accuract', accuracy/max_range, " | Total Time: ", stop-start, " | Time by single inference : ", (stop-start)/(max_range-min_range))
engine.destroy()