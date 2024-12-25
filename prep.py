import os
import argparse
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
def save_dataset(args):
    if not os.path.exists(os.path.join(args.save_path,"trainA")):
        os.makedirs(os.path.join(args.save_path,"trainA"))
    if not os.path.exists(os.path.join(args.save_path,"trainB")):
        os.makedirs(os.path.join(args.save_path,"trainB"))

    # patients_list = sorted([d for d in os.listdir(args.data_path) if 'zip' not in d])
    patients_list = os.listdir(args.data_path)
    for p_ind, patient in enumerate(patients_list):
        # patient_input_path = os.path.join(args.data_path, patient,
        #                                   "quarter_{}mm".format(args.mm))
        # patient_target_path = os.path.join(args.data_path, patient,
        #                                    "full_{}mm".format(args.mm))
        # print(patient_input_path)

        #if not os.path.exists(os.path.join(args.save_path, "trainA",patient)): #test_img
        #    os.makedirs(os.path.join(args.save_path, "trainA",patient))
        #if not os.path.exists(os.path.join(args.save_path, "trainB",patient)):
        #    os.makedirs(os.path.join(args.save_path, "trainB",patient))
        patient_input_path = args.data_path+ "/" +patient+ "/" +'quarter'  #train
        patient_target_path = args.data_path+ "/" +patient+ "/" +'full'     #train

        #patient_input_path = args.data_path+ "/" +patient+ "/" +'1mm'  #test_img
        #patient_target_path = args.data_path+ "/" +patient+ "/" +'3mm'     #test_img
        for path_ in [patient_input_path, patient_target_path]:
            full_pixels = get_pixels_hu(load_scan(path_)) #change by gzl
            #full_pixels = loaddata(path_)
            for pi in range(len(full_pixels)):
                io = 'trainA' if 'quarter' in path_ else 'trainB'  #input 改成trainA  target改成trainB
                #io = 'trainA' if '1' in path_
                #ima = normalize_(full_pixels[pi], args.norm_range_min, args.norm_range_max)
                f_name = '{}_{}.png'.format(patient, pi) #train
                #f_name = '{}.png'.format(pi) #test
                #np.save(os.path.join(args.save_path, f_name), f)
                #np.save(os.path.join(args.save_path,io,f_name),full_pixels[pi])
                #print(full_pixels)
                ima = setDicomWinWidthWinCenter(full_pixels[pi])
                #ima = setDicomWinWidthWinCenter(f)
                #print(full_pixels[pi])

                cv2.imwrite(os.path.join(args.save_path, io,f_name), ima)  #train
                #print(os.path.join(args.save_path,io,patient,f_name))
        printProgressBar(p_ind, len(patients_list),
                         prefix="save image ..",
                         suffix='Complete', length=25)
        print(' ')
def convert_from_dicom_to_jpg(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window*1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])    #归一化
    newimg = (newimg*255).astype('uint8')                #将像素值扩展到[0,255]
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def loaddata(path):
    slices = []
    for s in os.listdir(path):
        slices.append(pydicom.dcmread(os.path.join(path,s),force=True).pixel_array)
    return slices
def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    dicom_list = os.listdir(path)
    #print(dicom_list)
    # 给dicom文件排序
    sort_dic_num_first = []
    for dicom in dicom_list:
        sort_dic_num_first.append(int(dicom.split(".")[1]))
    sort_dic_num_first.sort()
    sorted_dicom_file = []
    for sort_num in sort_dic_num_first:
        for file in dicom_list:
            if str(sort_num) == file.split(".")[1]:
                sorted_dicom_file.append(file)
    slices = [pydicom.read_file(os.path.join(path, s)) for s in sorted_dicom_file]
    for s in sorted_dicom_file:
        print(os.path.join(path, s))
    #slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        #print("intercept:",intercept)
        #print("slope:",slope)
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

def setDicomWinWidthWinCenter(img_data, winwidth=300, wincenter=40, rows=512, cols=512):  #good
    # 腹部检查常设定窗宽为300 Hu~500 Hu,窗位30 Hu~50 Hu,肝脾CT检查应适当变窄窗宽以便
    # 更好发现病灶，窗宽为100 Hu~200 Hu,窗位为30 Hu~45 Hu,
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)
    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255
    return img_temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./datasets/D_exam')
    parser.add_argument('--save_path', type=str, default='./datasets/D_examimg')

    parser.add_argument('--test_patient', type=str, default='L01')
    #parser.add_argument('--mm', type=int, default=3)
    #parser.add_argument('--mm', type=str, default='DICOM_CT_PD')

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    args = parser.parse_args()

    save_dataset(args) #dicom数据提出numpy

    #测试AAPM数据集中IMA格式数据,IMA格式数据可以当作dicom格式操作
    #first_patient = load_scan(r"./datasets/AAPM/L067/full_1mm")
    #first_patient_pixels = get_pixels_hu(first_patient)
    #ima = setDicomWinWidthWinCenter(first_patient_pixels[50],400,40,512,512)

    #ima = pydicom.dcmread(r"./datasets/AAPM/L067/full_1mm/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA").pixel_array.data
    #ima = np.load(r"./datasets/dicom_npys/testA/P01_0.npy")
    #print(ima.dtype)
    #im = np.load(r"./results/dicom1_cyclegan/test_latest/npy_images/P01_0_fake_B.npy")
    #print(im.dtype)
    #im = np.load(r"./datasets/dicom_npys/trainB/P01_0.npy")
    #im = setDicomWinWidthWinCenter(img_data=ima)
    #im = Image.fromarray(np.load(r"./datasets/dicom_npys/trainB/P01_0.npy"),mode="L")
    #ima = setDicomWinWidthWinCenter(ima)
    #plt.imshow(ima, cmap=plt.cm.gray)
    #plt.imshow(im)
    #plt.show()
    #file = get_pixels_hu(load_scan("./test"))
    #ima = setDicomWinWidthWinCenter(file[0])
    #cv2.imwrite("./",ima)


