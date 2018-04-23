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

#For int8 
import caliberator
import glob
# Initilisations

MEAN = (71.60167789, 82.09696889, 72.30508881)
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
CALIBRATION_DATASET_LOC = '/home/d/Desktop/tensorRT/mnist/fccBased/caliberation_data_set/*.png'


def sub_mean_chw(data):
  # data = data.transpose((1,2,0)) # CHW -> HWC
  # data -= np.array(MEAN) # Broadcast subtract
  data = data.transpose((2,0,1)) # HWC -> CHW
  return data
             
def create_calibration_dataset():
  calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
  return calibration_files[:9]


calibration_files = create_calibration_dataset()
batchstream = caliberator.ImageBatchStream(5, calibration_files, sub_mean_chw)
int8_calibrator = caliberator.PythonEntropyCalibrator(["data"], batchstream)
# for i in range(0, len(calibration_files)):
# 	print(calibration_files[i])
# Creating Engine : By parsing .caffemodel and .prototxt
engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
                                       MODEL_PROTOTXT,
                                       CAFFE_MODEL,
                                       1,
                                       1 << 20,
                                       OUTPUT_LAYERS,
                                       trt.infer.DataType.FLOAT)

if(len(sys.argv) == 2):
	im = Image.open(sys.argv[1])
	imshow(np.asarray(im))
	arr = np.array(im)
	img = arr.ravel()
else:
	# Test Image
	path = 'data/zero.png'
	im = Image.open(path)
	imshow(np.asarray(im))
	arr = np.array(im)
	img = arr.ravel()
 
# Pre-Process image with the mean image 
parser = parsers.caffeparser.create_caffe_parser()
mean_blob = parser.parse_binary_proto(IMAGE_MEAN)
parser.destroy()
#NOTE: This is different than the C++ API, you must provide the size of the data
mean = mean_blob.get_data(INPUT_W ** 2)
data = np.empty([INPUT_W ** 2])
for i in range(INPUT_W ** 2):
    data[i] = float(img[i]) - mean[i]
mean_blob.destroy()

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

print ("Prediction: " + str(np.argmax(output)))
trt.utils.write_engine_to_file("mnsit_float32.engine", engine.serialize())
# new_engine = trt.utils.load_engine(G_LOGGER, "new_mnist.engine")

context.destroy()
engine.destroy()
# new_engine.destroy()
runtime.destroy()
