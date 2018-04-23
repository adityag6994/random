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
import time

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

CHANNEL = 1
HEIGHT  = 28
WIDTH   = 28


# File Path : Path to prototxt and weight files
MODEL_PROTOTXT = 'mnist_model.prototxt'
CAFFE_MODEL = 'mnist_model.caffemodel'
DATA = 'data/*.png'
IMAGE_MEAN = 'mnist_model.binaryproto'
CALIBRATION_DATASET_LOC = '/home/d/Desktop/tensorRT/mnist/fccBased/caliberation_data_set/*.png'
TEST_IMAGE = '/home/d/Desktop/tensorRT/mnist/fccBased/data/'
IMAGE_MEAN_NPY = 'mnist_model.npy'
RESULT_FILE = 'mnist_test_images.txt'

f  = open(RESULT_FILE, "r")
fs = f.read()
words  = fs.split()
number = [int(w) for w in words]

def sub_mean_chw(data):
  # data = data.transpose((1,2,0)) # CHW -> HWC
  mean_image = np.load(IMAGE_MEAN_NPY)
  mean_image = np.float32(mean_image)
  data -= np.array(mean_image) # Broadcast subtract
  # data = data.transpose((2,0,1)) # HWC -> CHW
  return data
             
def create_calibration_dataset():
  # calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
  calibration_files = glob.glob(DATA)
  return calibration_files[:20]


calibration_files = create_calibration_dataset()
batchstream = caliberator.ImageBatchStream(5, calibration_files, sub_mean_chw)
int8_calibrator = caliberator.PythonEntropyCalibrator(["data"], batchstream)
# caliberator.PythonEntropyCalibrator.write_calibration_cache(1,1000)
# for i in range(0, len(calibration_files)):
# 	print(calibration_files[i])
# Creating Engine : By parsing .caffemodel and .prototxt
engine = trt.lite.Engine(framework="c1",
                         deployfile=MODEL_PROTOTXT,
                         modelfile=CAFFE_MODEL,
                         max_batch_size=1,
                         max_workspace_size=(256 << 20),
                         input_nodes={"data":(CHANNEL,HEIGHT,WIDTH)},
                         output_nodes=["loss"],
                         preprocessors={"data":sub_mean_chw},               
                         calibrator=int8_calibrator,
                         logger_severity=trt.infer.LogSeverity.INFO)


start = time.clock()

accuracy = 0
MAX_COUNT = 1000
for i in range(0,MAX_COUNT):
	test_data = caliberator.ImageBatchStream.read_image_chw(TEST_IMAGE + str(i) + '.png')
	out = engine.infer(test_data)[0]
	print(np.argmax(out) ,' ', number[i])
	if(np.argmax(out) == number[i]):
		accuracy  = accuracy + 1
# for i in range(0, len(out)):
# 	print(i,' => ', out[i])

stop = time.clock()
print('inference time : ', (stop-start)/MAX_COUNT, ' | Total Time : ',(stop - start) , ' | Accuracy : ', accuracy/MAX_COUNT)
