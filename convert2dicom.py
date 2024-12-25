import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import cv2
if __name__ == "__main__":
    img = Image.open(r".\datasets\dicom80kv5mmimg\trainA\P01_38.png")
    #print(img.shape)
    a=np.ones((512,512))*1024
    nparray = np.array(img)
    data16 = np.int16(nparray)
    for i in data16:
        i =i/256.0*300-110
        print(i)
    data16=np.int16(data16)
   # plt.imshow(nparray,"gray")
    #plt.show()
    print(nparray.shape)
    print(nparray)

    '''data2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)'''
    ds = pydicom.dcmread(r".\datasets\dicom80kv5mm\P01\full\2.1.dcm")
    #ds.pixel_array.data = data16.tostring()
    #ds.pixel_array.data = data16
    ds.PixelData = data16.tobytes()
    #ds.Rows,ds.Columns = data2.shape
    ds.save_as("testdicom.dcm")