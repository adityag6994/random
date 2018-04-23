import caffe
import lmdb
import sys
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image
import io

if len(sys.argv) != 2:
    print ("Usage: python convert_mat_to_lmdb")


lmdb_env = lmdb.open(sys.argv[1])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()
path = 'data/'
count = 0
f = open('mnist_test_images.txt','w')
# f.write(str('a'))
# f.write(str('\n'))
# exit()
for key, value in lmdb_cursor:

    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    # print(type(data),' ', data[0,:,:].shape)
    im = Image.fromarray(data[0,:,:])
    #path = 'data/' + str(count) + '.png'
    #im.save(path)
    f.write(str(label))
    f.write('\n')
    count = count + 1	
    print('{},{},{}'.format(count,key, label), )


# import caffe
# import lmdb
# import sys
# import numpy as np
# from caffe.proto import caffe_pb2

# if len(sys.argv) != 2:
#     print ("Usage: python convert_mat_to_lmdb")


# lmdb_env = lmdb.open(sys.argv[1])
# lmdb_txn = lmdb_env.begin()
# lmdb_cursor = lmdb_txn.cursor()
# datum = caffe_pb2.Datum()

# for key, value in lmdb_cursor:
#     datum.ParseFromString(value)
#     label = datum.label
#     data = caffe.io.datum_to_array(datum)
#     print('{},{}'.format(key, label))
