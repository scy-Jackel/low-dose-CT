# low-dose-CT
 
prep.py processes the dicom data first, converts it into images and stores it in datasets

train.py is training. The training parameters are in options train_options and base_options python train.py --model --datasets ...

A checkpoints folder will be generated during training, which contains the generated model

datasets is the dataset directory. Import the data into this directory. Each dataset has four folders trainA corresponds to the training sample low-dose CT, trainB corresponds to the training sample standard measurement CT,
testA, testB correspond to the test sample low-dose CT, test sample standard measurement CT,

The models folder stores the model code

test.py is testing. The test parameters are in options test_options and base_options python test.py --model --datasets ...

Executing test.py will generate a results folder. The name of the generated result corresponds to checkpoints

util is responsible for visualization

convert2dicom is to convert the result into dicom
