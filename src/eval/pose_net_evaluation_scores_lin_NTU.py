from __future__ import print_function
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="7"
sys.path.append('/home/kha4hi/codes/Keras_projects/utils')
import fileinput
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import keras
from keras.models import load_model,Model
from keras.layers import Input
import h5py
import numpy as np
import xlsxwriter
import copy
from sklearn.metrics import classification_report

def line2rec_label(line):
    items = line.rsplit(None, 1)
    item1 = items[0]
    item2 = int(items[1])

    return item1, item2

test_file = \
"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_kinect/cross_subject/test/test_shifted_commonrotated/test_16487pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_kinect/cross_view/camera1/test_shifted_commonrotated/test_18932pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/2DPose_kinect/cross_subject/test/test_shifted/test_16487pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/2DPose_kinect/cross_view/camera1/test_shifted/test_18932pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_subject/test/test_bbshift_shifted_commonrotated/test_16560pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_view/camera1/test_bbshift_shifted_commonrotated/test_18960pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_subject/test/test_bbshift_2D_shifted/test_16560pos.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_view/camera1/test_bbshift_2D_shifted/test_18960pos.h5"
    #'/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_subject/test/test_shifted_commonrotated/test_16560pos.h5'
#model_name = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose3D_subJHMDB/subsplit2_8_16_256_pretrained_aug_67.50.h5'
model_name = \
"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose_kinect_NTU/cross_subject_kinect3D_shifted_commonrotated_aug3_77.75.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose_kinect_NTU/cross_view_kinect3D_shifted_commonrotated_aug3_88.53.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose_kinect_NTU/cross_subject_kinect2D_shifted_75.20.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose_kinect_NTU/cross_view_kinect2D_shifted_80.38.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose3D_RGB_NTU/cross_subject/64_128_1024_bbshift_shifted_commonrotated_aug3_80.80.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose3D_RGB_NTU/cross_view/64_128_1024_bbshift_shifted_commonrotated_aug3_91.68.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose2D_RGB_NTU/cross_subject/64_128_1024_bbshift2D_shifted_80.51.h5"
#"/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose2D_RGB_NTU/cross_view/64_128_1024_bbshift2D_shifted_89.11.h5"
    #'/mnt/Projects/CV-008_Students/ActionRecognitionProjects/Models/pose3D_RGB_NTU/cross_subject/64_128_1024_47nodes_cross_subject_shifted_commonrotated_79.92.h5'
workbook_name = '/home/lin7lr/temporal-segment-networks-master/output/NTU_Confusion_Matrix/NTU_kinect_3D/NTU_pose_Confusion_Matrix_cross_subject_kinect3D_shifted_commonrotated_aug3_77.75.xlsx'

#score_path = '/home/lin7lr/temporal-segment-networks-master/scores/jhmdb_subsplit_2_pose'
score_path = '/home/lin7lr/temporal-segment-networks-master/scores/NTU_pose_cross_subject_kinect3D_shifted_commonrotated_aug3_77.75'

save_scores = True
model = load_model(model_name)
print(model.summary())

with h5py.File(test_file, 'r') as f:
    x_test = f['/dataset'][()]  # (83, 3, 34, 10)
    y_test = f['/label'][()]
    y_test = y_test.T  # (83,1)
max_label = max(y_test)
num_classes = int(max_label[0])
y_test -= 1
#  (85, 3, 34, 10) ->   (85, 10, 34, 3)
x_test = np.transpose(x_test, (0, 3, 2, 1))
y_test_origin = copy.deepcopy(y_test)
y_test= keras.utils.to_categorical(y_test, num_classes)  # (83,2)

score_total = model.evaluate(x_test, y_test, verbose=1)  # consider the x_test as one single batch
scores = model.predict(x_test, verbose=0)
print('Test Loss :', score_total[0])
print('Test Accuracy:', score_total[1])
y_pred = model.predict_classes(x_test)
print(classification_report(y_test_origin, y_pred))

if save_scores is True:
    np.savez(score_path, scores=scores, labels=y_test_origin)


cf = confusion_matrix(y_test_origin,y_pred).astype(float)
print(cf)
acc = accuracy_score(y_test_origin, y_pred)
print(acc)
acc1 = acc*100

###Workbook Writing#####

workbook = xlsxwriter.Workbook(workbook_name)
worksheet = workbook.add_worksheet()

col = 0
row = 0

for  row, data in enumerate(cf):
    worksheet.write_row(row, col, data)
    row +=1
worksheet.write(row, 0, 'Overall_Accuracy')
worksheet.write(row, 1, acc1)

# row+=2
# worksheet.write(row, 0, 'Original_Label')
# worksheet.write(row, 1, 'Original_Video_label')
# worksheet.write(row, 2, 'Predicted_Label')
# worksheet.write(row, 3, 'Video_Name')
#
# row+=1
# start_row=row
# col=0
# for   data in (false_label_org):
#     worksheet.write(row, col, data)
#     row +=1
#
# row=start_row
# col=1
# for   data in (false_video_label):
#     worksheet.write(row, col, data)
#     row +=1
#
# row=start_row
# col=2
# for   data in (false_label_pred):
#     worksheet.write(row, col, data)
#     row +=1
#
# row=start_row
# col=3
# for   data in (false_video_name):
#     worksheet.write(row, col, data)
#     row +=1
workbook.close()
