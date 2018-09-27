from __future__ import print_function
import scipy.io as sio
import os, sys, shutil
#os.environ["CUDA_VISIBLE_DEVICES"]="5"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import keras
from keras.callbacks import ModelCheckpoint,LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import load_model,Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.initializers import glorot_normal

import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
import copy


def posenetUsman_NTU(num_classes, input_shape, params_dict):
    alpha_leakyRelu = 0.5

    batchNormFlag = params_dict['batchNormFlag']
    FCDropout = params_dict['FCDropout']
    numKernelConv2D1 = params_dict['numKernelConv2D1']
    numKernelConv2D2 = params_dict['numKernelConv2D2']
    sizeDense = params_dict['sizeDense']
    xavierInitFlag = params_dict['xavierInitFlag']
    dropoutAfterConvFlag = params_dict['dropoutAfterConvFlag']
    dropoutAfterConv = params_dict['dropoutAfterConv']

    model = Sequential()
    if xavierInitFlag == True:
        model.add(Conv2D(numKernelConv2D1, (3, 3), activation='linear', input_shape=input_shape,
                         padding='same', name='conv2d_1', kernel_initializer=glorot_normal(seed=None)))
    else:
        model.add(Conv2D(numKernelConv2D1, (3, 3), activation='linear', input_shape=input_shape, padding='same',
                         name='conv2d_1'))
    if batchNormFlag == True:
        model.add(BatchNormalization(axis=-1, name='batchnorm_1'))

    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=alpha_leakyRelu))
    #model.add(PReLU())
    if dropoutAfterConvFlag == True:
        model.add(Dropout(dropoutAfterConv))
    print('input shape', input_shape)
    print(model.output_shape)  # (None, 10, 34, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    print(model.output_shape) # (None, 9, 33, 64)



    if xavierInitFlag == True:
        model.add(Conv2D(numKernelConv2D2, (3, 3), activation='linear', name='conv2d_2', kernel_initializer=glorot_normal(seed=None)))
    else:
        model.add(Conv2D(numKernelConv2D2, (3, 3), activation='linear', name='conv2d_2'))
    if batchNormFlag == True:
        model.add(BatchNormalization(axis=-1, name='batchnorm_2'))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=alpha_leakyRelu))
    #model.add(PReLU())
    if dropoutAfterConvFlag == True:
        model.add(Dropout(dropoutAfterConv))
    print(model.output_shape) # (None, 7, 32, 128)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Flatten())


    if xavierInitFlag == True:
        model.add(Dense(sizeDense, activation='linear', name='dense_1_pennaction', kernel_initializer= glorot_normal(seed=None)))
    else:
        model.add(Dense(sizeDense, activation='linear', name='dense_1_pennaction'))
    if batchNormFlag == True:
        model.add(BatchNormalization(name='batchnorm_3_pennaction'))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=alpha_leakyRelu))
    #model.add(PReLU())
    model.add(Dropout(FCDropout))   # fraction of the input units to drop

    if xavierInitFlag == True:
        model.add(Dense(num_classes, activation='softmax', name='dense_2_pennaction', kernel_initializer= glorot_normal(seed=None)))
    else:
        model.add(Dense(num_classes, activation='softmax', name='dense_2_pennaction'))


    print(model.summary())
    return model

def posenetUsman_preTrained(num_classes, input_shape, preTrained_model):
    model = posenetUsman(num_classes, input_shape)
    model.load_weights(preTrained_model, by_name=True)
    return model

def posenetUsman1(num_classes, input_shape ):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))

    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(Dropout(0.6))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    return model

def posenetUsman2(num_classes, input_shape ):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))

    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.6))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    return model

def posenetLin1(num_classes, input_shape ):
    dropout_rate = 0.6
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))

    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    # print(model.output_shape) # (None, 9, 33, 64)
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # print(model.output_shape) # (None, 7, 32, 128)
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(dropout_rate))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    return model

def posenetLin2(num_classes, input_shape ):
    dropout_rate = 0.6
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))

    # # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    # # print(model.output_shape) # (None, 9, 33, 64)
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # # print(model.output_shape) # (None, 7, 32, 128)
    # model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(dropout_rate))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    return model

def posenetLin(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 2), activation='relu', input_shape=input_shape, padding='same'))   # (None, 10, 34, 3)  ->  (None, 10, 34, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))          # (None, 10, 34, 64)  ->  (None, 9, 33, 64)
    model.add(Conv2D(64, (3, 2), activation='relu', padding='same'))  #  (None, 9, 33, 64)  ->  (None, 9, 33, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))       # (None, 9, 33, 64)  ->  (None, 8, 32, 64)
    model.add(Conv2D(128, (3, 2), activation='relu', padding='same'))   # (None, 8, 32, 64)  ->  (None, 8, 32, 128)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))  #  (None, 8, 32, 128) -> (None, 7, 31, 128)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))


def print_result_2D(model, x_test, y_test, y_test_origin, videoname, modelType):
    score1 = model.evaluate(x_test, y_test, verbose=1, batch_size= batch_size)
    print(modelType + ' model ' + videoname + ' Test Accuracy:', score1[1])
    y_pred = model.predict_classes(x_test)
    print(classification_report(y_test_origin, y_pred))

def print_result_3D(model, x_test, y_test, y_test_origin,  videoname, modelType):
    score1 = model.evaluate(x_test, y_test, verbose=1, batch_size = batch_size)  # consider the x_test as one single batch, len() : number of rows
    print(modelType + ' model '+ videoname +' Test Accuracy:', score1[1])
    y_pred = model.predict_classes(x_test)
    print(classification_report(y_test_origin, y_pred))

    # gt_pred = []
    # for test_sampleid in range(len(x_test)):
    #     gt_pred.append( (y_test_origin[test_sampleid, 0], y_pred[test_sampleid], test_videoname_list[test_sampleid]) )
    #
    # gt_pred_sorted = sorted(gt_pred, key= lambda x:(x[0], x[1]))
    # for (y_gt, y_pred, videoname) in gt_pred_sorted:
    #     print('y_gt: ', int(y_gt), ', y_pred: ', y_pred, ' video_name: ', videoname)
    return y_pred

def line2rec_label(line):
    items = line.rsplit(None, 1)
    item1 = items[0]
    item2 = int(items[1])

    return item1, item2
def scheduler(epoch):
    if epoch < epoch1:
        #K.set_value(model.optimizer.lr, float(learning_rate2))
        return float(learning_rate1)
    elif epoch >=epoch1 and epoch < epoch2:
        print("learning rate changed into ", learning_rate2)
        return float(learning_rate2)
    else:
        print("learning rate changed into ", learning_rate3)
        return float(learning_rate3)
if __name__ == '__main__':
    train_file = \
    "/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_subject/train/train_bbshift_shifted_commonrotated/train_120960pos_Aug3.h5"

    # val_file = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose/cross_view/camera_1/test_47node_resize_rotation/test_18729pos.h5'
    test_file = \
    "/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/h5/3DPose_RGB/cross_subject/test/test_bbshift_shifted_commonrotated/test_16560pos.h5"


    num_classes = 60
    batch_size = 15
    num_epochs = 100
    learning_rate1 = 0.0001
    epoch1 = 5
    learning_rate2 = 0.00005
    epoch2 = 10
    learning_rate3 = 0.00001
    # learning_rate1 = 0.00005
    # epoch1 = 10
    # learning_rate2 = 0.00005
    # epoch2 = 20
    # learning_rate3 = 0.00001

    params_dict = dict()
    params_dict['batchNormFlag'] = True

    params_dict['FCDropout'] = 0.5

    params_dict['numKernelConv2D1'] = 128
    params_dict['numKernelConv2D2'] = 256
    params_dict['sizeDense'] = 2048

    params_dict['xavierInitFlag'] = True

    params_dict['dropoutAfterConvFlag'] = False
    params_dict['dropoutAfterConv'] = 0.1

    pose_type = '3D'   # '2D'


    preTrained = False
    use_saved_init = False

    saved_init_model_to_load = '/home/lin7lr/test_pose/preTrainedModel/Penn_action/model_and_weights_randm_init_8_16_256.h5'
    preTrained_model = './models/best_model_jhmdb/jhmdb_subsplit1_76_lessLoss.h5'
    random_init_path_to_save = "/home/lin7lr/test_pose/TrainingModelLog/NTU/model_and_weights_randm_init.h5"

    saved_model_path = "/home/lin7lr/test_pose/TrainingModelLog/NTU/model_and_weights_64_128_1024_47nodes_60classes.h5"  #######################################################################
    tensorBoardDir = '/home/lin7lr/test_pose/TrainingModelLog/NTU/logs'  ###########################################################################################

    # test_txt_data = [line2rec_label(x) for x in
    #                  open('/home/lin7lr/test_pose/list/Penn_action/rgb_testlist.txt')]
    #
    # # label_list = [line2rec_label(x) for x in open('/home/lin7lr/test_pose/list/JHMDB/train_test_subsplit1/class_sublist.txt')]
    # test_videoname_list = []
    # for line_id in range(len(test_txt_data)):
    #     rgb_path = test_txt_data[line_id][0]
    #     rgb_path_split = rgb_path.split('/')
    #     test_videoname_list.append(rgb_path_split[-2])
    with h5py.File(train_file, 'r') as f:
        x_train = f['/dataset'][()]  # (1038, 3, 34, 10)
        y_train = f['/label'][()]
        y_train = y_train.T  # (1038,1), array

    shuffled_order = list(range(x_train.shape[0]))
    random.shuffle(shuffled_order)
    x_train = x_train[shuffled_order , :, :, : ]
    y_train = y_train[shuffled_order]

    # with h5py.File(val_file, 'r') as f:
    #     x_val = f['/dataset'][()]
    #     y_val = f['/label'][()]
    #     y_val = y_val.T  # (83,1)

    with h5py.File(test_file, 'r') as f:
        x_test = f['/dataset'][()]   #  (83, 3, 34, 10)
        y_test = f['/label'][()]
        y_test = y_test.T  # (83,1)

    dim = x_train.shape  # (1038, 3, 34, 10)
    # label = max(y_train)  # 4, label is a vector
    print("x_train shape: ", dim)

    img_rows = dim[3] # 10
    img_cols = dim[2] # 34
    channels = dim[1] # 3
    input_shape=(img_rows,img_cols,channels)
    # y_train is a (133,1) array, y_test is a (31,1) array

    ######################
    y_train -= 1   # to make the labels 0-based instead of 1-based
    # y_val -= 1
    y_test -= 1

    # (num_samples, 3, img_cols, 10) -> # (num_samples, 10, img_cols, 3)
    # numpy.transpose can be used here !!!!

   # x_train_reshape = x_train.reshape(x_train.shape[0],  img_rows, img_cols, channels)
   # x_train_transpose = np.transpose(x_train, (0, 3, 2,1))




   # x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, channels)
    x_train = np.transpose( x_train,(0, 3, 2,1) )
    #  (85, 3, 34, 10) ->   (85, 10, 34, 3)
    # x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)

   # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    x_test = np.transpose(x_test, (0, 3, 2,1))
    # y_train_origin = copy.deepcopy(y_train)
    # y_val_origin = copy.deepcopy(y_val)
    y_test_origin = copy.deepcopy(y_test)


    y_train = keras.utils.to_categorical(y_train, num_classes)   # (1038,2)
    # y_val = keras.utils.to_categorical(y_val, num_classes)  # (83,2)
    y_test= keras.utils.to_categorical(y_test, num_classes)  # (83,2)


    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')

    # netName = sys.argv[1]
    netName = 'posenet'

    if netName == 'posenet':
        if preTrained == False:
            model = posenetUsman_NTU(num_classes=num_classes, input_shape=input_shape, params_dict=params_dict)
            if use_saved_init == True:
                model.load_weights(saved_init_model_to_load)
            else:  # if the saved initialization is not to be used,  save the new initialization
                try:
                    os.remove(random_init_path_to_save)
                except OSError:
                    pass
                model.save_weights(random_init_path_to_save)
        else: # pretrained case
            # model = Sequential()
            # model.add(Conv2D(64, (3, 2), activation='relu', input_shape=input_shape))
            # print(model.output_shape)
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
            # print(model.output_shape)
            # model.add(Conv2D(128, (3, 2), activation='relu'))
            # print(model.output_shape)
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
            # model.add(Flatten())
            # model.add(Dense(1024, activation='relu',name='Dense1'))
            # #model.add(Dropout(0.8,name='Dropout'))
            # model.add(Dense(num_classes, activation='softmax',name='SoftMax'))
            # model.load_weights(preTrained_model, by_name=True)
            # print(model.summary())
            # #model = load_model(preTrained_model)
            model = posenetUsman_preTrained(num_classes=num_classes, input_shape=input_shape,
                                            preTrained_model=preTrained_model)

        try:
            os.remove(saved_model_path)
        except OSError:
            pass


        try:
            shutil.rmtree(tensorBoardDir)
        except Exception:
            pass
    else:
        raise ValueError("Invalid model name!")

    # ModelCheckpoint :  callback function ,  save the model after every epoch
    checkpointCallback = ModelCheckpoint(saved_model_path, monitor='val_acc', verbose=0, save_best_only=True,
                                         save_weights_only=False, mode='auto', period=1)
    # TensorBoard : callback function, TensorBoard basic visualizations,  a visualization provided with Tensorflow
    # command : tensorboard --logdir=/full_path_to_your_logs
    tensorBoardCallback = keras.callbacks.TensorBoard(log_dir=tensorBoardDir, histogram_freq=0, batch_size=batch_size,
                  write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    changelr_callback = LearningRateScheduler(scheduler)
    # before training a model, we need to configure the learning process
    # loss function: the objective that the model will try to minimize
    # metrics: For any classification problem you will want to set this to metrics = ['accuracy']

    if learning_rate1 is not None:
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=learning_rate1,
                                         beta_1=0.9, beta_2= 0.999, epsilon=1e-08, decay=0.0),metrics=['accuracy'])
    else:
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    # fit : trains the model for a fixed number f epochs
    # returns a Histroy object, Its History.history attribute is a record of training loss values and metrics values at successive epochs,
    # as well as validation loss values and validation metrics values (if applicable).
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.05,
                        callbacks=[checkpointCallback, tensorBoardCallback,early_stopping_callback, changelr_callback])


    # history = model.fit(x_train,y_train,batch_size=batch_size,epochs=num_epochs, verbose=1, validation_split= 0.1, validation_data=None,callbacks=[checkpointCallback, tensorBoardCallback])
    if pose_type == '3D':
        y_pred = print_result_3D(model, x_test, y_test, y_test_origin,  'NTU', 'final')
    elif pose_type == '2D':
        print_result_2D(model, x_test, y_test, y_test_origin,  '5734_5735', 'final')
        print_result_2D(model, x_test2, y_test2, y_test2_origin, 'Laptop6', 'final')
        print_result_2D(model, x_test3, y_test3, y_test3_origin, 'Laptop5', 'final')
        print_result_2D(model, x_test4, y_test4, y_test4_origin, 'Laptop1', 'final')
        # print_result_2D(model, x_test, y_test, y_test_origin, videoname, modelType)
    else:
        raise ValueError("Invalid pose type!")

    model_best_acc = load_model(saved_model_path)
    if pose_type == '3D':
        y_pred_best_acc = print_result_3D(model_best_acc, x_test, y_test, y_test_origin,'NTU', 'best acc')
    elif pose_type == '2D':
        print_result_2D(model_best_acc, x_test, y_test, y_test_origin, '5734_5735', 'best acc')
        print_result_2D(model_best_acc, x_test2, y_test2, y_test2_origin, 'Laptop6', 'best acc')
        print_result_2D(model_best_acc, x_test3, y_test3, y_test3_origin, 'Laptop5', 'best acc')
        print_result_2D(model_best_acc, x_test4, y_test4, y_test4_origin, 'Laptop1', 'best acc')
    else:
        raise ValueError("Invalid pose type!")

    # summarize history for accuracy
    plt.figure(1)
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    if learning_rate1 is not None:
        plt.title('model accuracy lr='+str(learning_rate1) + ' batch ='+ str(batch_size))
    else:
        plt.title('model accuracy lr= Adadelta'+ ' batch ='+ str(batch_size))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid(True)

    # summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if learning_rate1 is not None:
        plt.title('model loss lr='+str(learning_rate1)+ ' batch ='+ str(batch_size))
    else:
        plt.title('model loss lr= Adadelta'+ ' batch ='+ str(batch_size))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid(True)
    plt.show()



    """
    plot confusion matrix
    """
    labels = list()
    str_labels = list()
    for i in range(num_classes):
        labels.append((i))
        str_labels.append(str(i))
    cm = confusion_matrix(y_test_origin, y_pred, labels)
    cm_best_acc = confusion_matrix(y_test_origin, y_pred_best_acc, labels)
    print('confusion matrix=')
    print(cm)
    print('confusion matrix best acc=')
    print(cm_best_acc)

    fig = plt.figure(2)
    ax1 = fig.add_subplot(121)
    cax1 = ax1.matshow(cm,cmap = plt.cm.Greys)
    plt.title('confusion matrix')
    fig.colorbar(cax1)
    ax1.set_xticks(labels)
    ax1.set_xticklabels(str_labels, fontsize=7)
    ax1.set_yticks(labels)
    ax1.set_yticklabels(str_labels)
    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(True)

    ax2 = fig.add_subplot(122)
    cax2 = ax2.matshow(cm_best_acc,cmap = plt.cm.Greys)
    plt.title('confusion matrix')
    fig.colorbar(cax2)
    ax2.set_xticks(labels)
    ax2.set_xticklabels(str_labels, fontsize=7)
    ax2.set_yticks(labels)
    ax2.set_yticklabels(str_labels)
    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(True)

    plt.show()