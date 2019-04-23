import numpy as np
from PIL import Image
import os
import scipy.io as scio

# Before running this code, the images must be adjusted to be RGB format.



fileOS = 'D:/cars_test/'
dataFile = 'D:/cars_test_annos_withlabels.mat'

# When preprossessing the training data, run these two line below.

# fileOS = 'D:/cars_train/'
# dataFile = 'D:/cars_train_annos.mat'


def get_data(datafile):

    data = scio.loadmat(datafile)
    anno = np.asarray(data['annotations'])
    y_train = int(anno['class'][0][0])
    x_1 = int(anno['bbox_x1'][0][0])
    y_1 = int(anno['bbox_y1'][0][0])
    x_2 = int(anno['bbox_x2'][0][0])
    y_2 = int(anno['bbox_y2'][0][0])

    for count in range(anno['class'].shape[1]-1):
        y_train = np.append(y_train,anno['class'][0][count+1])
        x_1 = np.append(x_1, anno['bbox_x1'][0][count + 1])
        y_1 = np.append(y_1, anno['bbox_y1'][0][count + 1])
        x_2 = np.append(x_2, anno['bbox_x2'][0][count + 1])
        y_2 = np.append(y_2, anno['bbox_y2'][0][count + 1])
    return x_1,y_1,x_2,y_2,y_train


def get_name(num):
    if num < 10:
        return '0000'+str(num)+'.jpg'
    elif num < 100:
        return '000'+str(num)+'.jpg'
    elif num < 1000:
        return '00'+str(num)+'.jpg'
    elif num < 10000:
        return '0'+str(num)+'.jpg'
    else:
        return str(num)+'.jpg'


def get_image_data(fileOS, x_1,y_1,x_2,y_2):
    i = 0
    imagelist = []
    for filename in os.listdir(fileOS):
        tempimage = Image.open(fileOS+filename)
        box = (x_1[i],y_1[i],x_2[i],y_2[i])
        print(i)
        tempimage = tempimage.crop(box)
        tempimage = tempimage.resize((224,224))
        r, g, b = tempimage.split()
        r = np.matrix(r)
        g = np.matrix(g)
        b = np.matrix(b)
        tempimage = np.asarray((r,g,b))
        imagelist.append(tempimage)
        i = i + 1
    imagelist = np.asarray(imagelist)
    return imagelist


x_1, y_1, x_2, y_2, y_test = get_data(dataFile)

X_test = get_image_data(fileOS,x_1,y_1,x_2,y_2)
print(X_test.shape)
print(X_test[0].shape)
print(y_test.shape)

np.save('d:/X_test',X_test)
np.save('d:/y_test',y_test)

# When saving the training data run these two line instead of those two line above.
# np.save('d:/X_test',X_test)
# np.save('d:/y_test',y_test)
