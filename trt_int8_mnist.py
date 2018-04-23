import glob
from random import shuffle
import numpy as np
from PIL import Image

import tensorrt as trt

# import labels        #from cityscapes evaluation script
import caliberator_int8    #calibrator.py

MEAN = (1, 1, 1)
MODEL_DIR = ''
CITYSCAPES_DIR = 'caliberation_data_set/'
TEST_IMAGE = CITYSCAPES_DIR + 'data/0.png'
CALIBRATION_DATASET_LOC = CITYSCAPES_DIR + '999*.png'

CLASSES = 10
CHANNEL = 1
HEIGHT = 28
WIDTH = 28

def sub_mean_chw(data):
  data = data.transpose((1,2,0)) # CHW -> HWC
  data -= np.array(MEAN) # Broadcast subtract
  data = data.transpose((2,0,1)) # HWC -> CHW
  return data
             
# def color_map(output):
#   output = output.reshape(CLASSES, HEIGHT, WIDTH)
#   out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
#   for x in range (WIDTH):
#     for y in range (HEIGHT):
#       out_col[y,x] = labels.id2label[labels.trainId2label[np.argmax(output[:,y,x])].id].color
#   return out_col

def create_calibration_dataset():
  # Create list of calibration images (filename)
  # This sample code picks 100 images at random from training set
  calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
  shuffle(calibration_files)
  return calibration_files[:100]

def main():
  calibration_files = create_calibration_dataset()
  
  # Process 5 images at a time for calibration
  # This batch size can be different from MaxBatchSize (1 in this example)
  batchstream = calibrator.ImageBatchStream(5, calibration_files, sub_mean_chw)
  int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)
  
  # Easy to use TensorRT lite package
  engine = trt.lite.Engine(framework="c1",
                           deployfile= "mnist_model.prototxt",
                           modelfile= "mnist_model.caffemodel",
                           max_batch_size=1,
                           max_workspace_size=(256 << 20),
                           input_nodes={"data":(CHANNEL,HEIGHT,WIDTH)},
                           output_nodes=["score"],
                           preprocessors={"data":sub_mean_chw},
                           postprocessors={"score":color_map},
                           data_type=trt.infer.DataType.INT8,
                           calibrator=int8_calibrator,
                           logger_severity=trt.infer.LogSeverity.INFO)
                           
  # test_data = calibrator.ImageBatchStream.read_image_chw(TEST_IMAGE)
  # out = engine.infer(test_data)[0]
  # test_img = Image.fromarray(out, 'RGB')
  # test_img.show()  

print('--------------------')
print('-- Double Complete--')
print('--------------------')
