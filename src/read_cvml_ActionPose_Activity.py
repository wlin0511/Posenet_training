from __future__ import print_function
import os, sys
import xml.etree.ElementTree as ET
import numpy as np
import scipy.io as sio
import math
import cv2 as cv
import json

class Box:
    def __init__(self):
        self.X=0
        self.Y=0
        self.Width=0
        self.Height=0

class Pose:
    def __init__(self):
        self.LAnkle_X = 0
        self.LAnkle_Y = 0
        self.LEar_X = 0
        self.LEar_Y = 0
        self.LElbow_X = 0
        self.LElbow_Y = 0
        self.LEye_X = 0
        self.LEye_Y = 0
        self.LHip_X = 0
        self.LHip_Y = 0
        self.LKnee_X = 0
        self.LKnee_Y = 0
        self.LShoulder_X = 0
        self.LShoulder_Y = 0
        self.LWrist_X = 0
        self.LWrist_Y = 0
        self.Neck_X = 0
        self.Neck_Y = 0
        self.Nose_X = 0
        self.Nose_Y = 0
        self.RAnkle_X = 0
        self.RAnkle_Y = 0
        self.REar_X = 0
        self.REar_Y = 0
        self.RElbow_X = 0
        self.RElbow_Y = 0
        self.REye_X = 0
        self.REye_Y = 0
        self.RHip_X = 0
        self.RHip_Y = 0
        self.RKnee_X = 0
        self.RKnee_Y = 0
        self.RShoulder_X = 0
        self.RShoulder_Y = 0
        self.RWrist_X = 0
        self.RWrist_Y = 0

def union_area(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return w*h

def intersection_area(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return 0
  return w*h

def get_object_data(object_id, root,maxFrame):  ####################################### GT cvml file,   read by object id
    objlist = []
    action_name = []
    boxObj = Box()
    poseObj = Pose()
    for child in root:
        frame=child.get('number')
        #print(frame)
        if child.tag == 'frame':
            for object in child:
                if object.tag == 'objectlist':
                    for objects in object:
                        if objects.tag == 'object':
                            if objects.get('id') == str(object_id) and int(frame) <= maxFrame:
                                #print('frame_num :', frame, object_id)
                                for box in objects:
                                    if box.tag == 'box':
                                        box_height = box.get('height')
                                        box_width = box.get('width')
                                        box_x = box.get('x')
                                        box_y = box.get('y')
                                        boxObj = Box()
                                        boxObj.X = box_x
                                        boxObj.Y = box_y
                                        boxObj.Width = box_width
                                        boxObj.Height = box_height
                                    if box.tag == 'personDetail': ################################ actually there is no personDetail in GT cvml file
                                        poseObj = Pose()
                                        for personDetail in box:
                                            joint_name = personDetail.get('Name')
                                            if joint_name == 'LAnkle':
                                                poseObj.LAnkle_X = personDetail.get('X')
                                                poseObj.LAnkle_Y = personDetail.get('Y')
                                            if joint_name == 'LEar':
                                                poseObj.LEar_X = personDetail.get('X')
                                                poseObj.LEar_Y = personDetail.get('Y')
                                            if joint_name == 'LElbow':
                                                poseObj.LElbow_X = personDetail.get('X')
                                                poseObj.LElbow_Y = personDetail.get('Y')
                                            if joint_name == 'LEye':
                                                poseObj.LEye_X = personDetail.get('X')
                                                poseObj.LEye_Y = personDetail.get('Y')
                                            if joint_name == 'LHip':
                                                poseObj.LHip_X = personDetail.get('X')
                                                poseObj.LHip_Y = personDetail.get('Y')
                                            if joint_name == 'LKnee':
                                                poseObj.LKnee_X = personDetail.get('X')
                                                poseObj.LKnee_Y = personDetail.get('Y')
                                            if joint_name == 'LShoulder':
                                                poseObj.LShoulder_X = personDetail.get('X')
                                                poseObj.LShoulder_Y = personDetail.get('Y')
                                            if joint_name == 'LWrist':
                                                poseObj.LWrist_X = personDetail.get('X')
                                                poseObj.LWrist_Y = personDetail.get('Y')
                                            if joint_name == 'Neck':
                                                poseObj.Neck_X = personDetail.get('X')
                                                poseObj.Neck_Y = personDetail.get('Y')
                                            if joint_name == 'Nose':
                                                poseObj.Nose_X = personDetail.get('X')
                                                poseObj.Nose_Y = personDetail.get('Y')
                                            if joint_name == 'RAnkle':
                                                poseObj.RAnkle_X = personDetail.get('X')
                                                poseObj.RAnkle_Y = personDetail.get('Y')
                                            if joint_name == 'REar':
                                                poseObj.REar_X = personDetail.get('X')
                                                poseObj.REar_Y = personDetail.get('Y')
                                            if joint_name == 'RElbow':
                                                poseObj.RElbow_X = personDetail.get('X')
                                                poseObj.RElbow_Y = personDetail.get('Y')
                                            if joint_name == 'REye':
                                                poseObj.REye_X = personDetail.get('X')
                                                poseObj.REye_Y = personDetail.get('Y')
                                            if joint_name == 'RHip':
                                                poseObj.RHip_X = personDetail.get('X')
                                                poseObj.RHip_Y = personDetail.get('Y')
                                            if joint_name == 'RKnee':
                                                poseObj.RKnee_X = personDetail.get('X')
                                                poseObj.RKnee_Y = personDetail.get('Y')
                                            if joint_name == 'RShoulder':
                                                poseObj.RShoulder_X = personDetail.get('X')
                                                poseObj.RShoulder_Y = personDetail.get('Y')
                                            if joint_name == 'RWrist':
                                                poseObj.RWrist_X = personDetail.get('X')
                                                poseObj.RWrist_Y = personDetail.get('Y')
                                    if box.tag == 'boolAttributes':
                                        action_name = []
                                        for labels in box:
                                            label_name = labels.get('name')
                                            label_value = labels.get('value')
                                            if label_value == 'true':
                                                action_name.append(label_name)
                                print('frame_num : ', frame, 'object_id :', object_id, 'action_name :', action_name , 'box_x :', boxObj.X , 'pose_neck :', poseObj.Neck_X)
                                objlist.append({'frame' : frame,'object_id' : object_id, 'box' : boxObj,  'action' : action_name, 'pose' : poseObj})
    return objlist

def get_pose_data(frameNum ,root):   ################################################################# pose cvml file,   read by frame
    objlist = []
    action_name = []
    boxObj = Box()
    poseObj = Pose()
    for child in root:
        frame=child.get('number')
        if child.tag == 'frame':
            if int(frame) == frameNum:
                for object in child:
                    if object.tag == 'objectlist':
                        for objects in object:
                            if objects.tag == 'object':
                                for box in objects:
                                    object_id= objects.get('id')
                                    if box.tag == 'box':
                                        box_height = box.get('height')
                                        box_width = box.get('width')
                                        box_x = box.get('x')
                                        box_y = box.get('y')
                                        boxObj = Box()
                                        boxObj.X = box_x
                                        boxObj.Y = box_y
                                        boxObj.Width = box_width
                                        boxObj.Height = box_height
                                    if box.tag == 'personDetail':
                                        poseObj = Pose()
                                        for personDetail in box:
                                            joint_name = personDetail.get('Name')
                                            if joint_name == 'LAnkle':
                                                poseObj.LAnkle_X = personDetail.get('X')
                                                poseObj.LAnkle_Y = personDetail.get('Y')
                                            if joint_name == 'LEar':
                                                poseObj.LEar_X = personDetail.get('X')
                                                poseObj.LEar_Y = personDetail.get('Y')
                                            if joint_name == 'LElbow':
                                                poseObj.LElbow_X = personDetail.get('X')
                                                poseObj.LElbow_Y = personDetail.get('Y')
                                            if joint_name == 'LEye':
                                                poseObj.LEye_X = personDetail.get('X')
                                                poseObj.LEye_Y = personDetail.get('Y')
                                            if joint_name == 'LHip':
                                                poseObj.LHip_X = personDetail.get('X')
                                                poseObj.LHip_Y = personDetail.get('Y')
                                            if joint_name == 'LKnee':
                                                poseObj.LKnee_X = personDetail.get('X')
                                                poseObj.LKnee_Y = personDetail.get('Y')
                                            if joint_name == 'LShoulder':
                                                poseObj.LShoulder_X = personDetail.get('X')
                                                poseObj.LShoulder_Y = personDetail.get('Y')
                                            if joint_name == 'LWrist':
                                                poseObj.LWrist_X = personDetail.get('X')
                                                poseObj.LWrist_Y = personDetail.get('Y')
                                            if joint_name == 'Neck':
                                                poseObj.Neck_X = personDetail.get('X')
                                                poseObj.Neck_Y = personDetail.get('Y')
                                            if joint_name == 'Nose':
                                                poseObj.Nose_X = personDetail.get('X')
                                                poseObj.Nose_Y = personDetail.get('Y')
                                            if joint_name == 'RAnkle':
                                                poseObj.RAnkle_X = personDetail.get('X')
                                                poseObj.RAnkle_Y = personDetail.get('Y')
                                            if joint_name == 'REar':
                                                poseObj.REar_X = personDetail.get('X')
                                                poseObj.REar_Y = personDetail.get('Y')
                                            if joint_name == 'RElbow':
                                                poseObj.RElbow_X = personDetail.get('X')
                                                poseObj.RElbow_Y = personDetail.get('Y')
                                            if joint_name == 'REye':
                                                poseObj.REye_X = personDetail.get('X')
                                                poseObj.REye_Y = personDetail.get('Y')
                                            if joint_name == 'RHip':
                                                poseObj.RHip_X = personDetail.get('X')
                                                poseObj.RHip_Y = personDetail.get('Y')
                                            if joint_name == 'RKnee':
                                                poseObj.RKnee_X = personDetail.get('X')
                                                poseObj.RKnee_Y = personDetail.get('Y')
                                            if joint_name == 'RShoulder':
                                                poseObj.RShoulder_X = personDetail.get('X')
                                                poseObj.RShoulder_Y = personDetail.get('Y')
                                            if joint_name == 'RWrist':
                                                poseObj.RWrist_X = personDetail.get('X')
                                                poseObj.RWrist_Y = personDetail.get('Y')
                                    if box.tag == 'boolAttributes':
                                        action_name = []
                                        for labels in box:
                                            label_name = labels.get('name')
                                            label_value = labels.get('value')
                                            if label_value == 'true':
                                                action_name.append(label_name)
                                print('frame_num : ', frame, 'object_id :', object_id, 'box_x :', boxObj.X , 'pose_neck :', poseObj.Neck_X)
                                objlist.append({'frame' : frame,'object_id' : object_id, 'box' : boxObj,  'action' : action_name, 'pose' : poseObj})
    return objlist

def get_positive_samples(objData,object_id,classes,min_samples,max_samples):
    numSamples = len(objData[object_id - 1])  # number of frames
    posSamples = []
    numClasses = len(classes)
    for classid in range(numClasses):
        sampleid = 0
        sampleid_ = 0
        while sampleid < numSamples:
            action = objData[object_id - 1][sampleid]['action']
            if classes[classid] in action:      # the first positive frame is found
                sampleid_ = sampleid
                sequence = []
                count = 0
                for sampleid_ in range(sampleid, numSamples):  # search from this frame to the end of this segment
                    action = objData[object_id - 1][sampleid_]['action']
                    if (len(action)!= 0) and (classes[classid] in action):  # if action is not empty and the class is found in action name set
                        frame = objData[object_id - 1][sampleid_]['frame']
                        box = objData[object_id - 1][sampleid_]['box']            ## do we need to check whether the bounding box exists ?????????
                        pose = objData[object_id - 1][sampleid_]['pose']         ## there are no pose data in GT cvml
                        action = classes[classid]
                        if count < max_samples:  # 240 frames
                            sequence.append({'frame' : frame,'object_id' : object_id, 'box' : box,  'action' : action, 'pose' : pose})

                    else:
                        # print('Break')
                        break
                    sampleid_ += 1
                    count += 1
                # end of 'for' : search from this frame to the end of this segment
                if count >= min_samples :
                    posSamples.append(sequence)   # add this sequence to positive sample set
                sampleid = sampleid_
            else:   # negative frame
                # print('Negative')
                sampleid += 1
                continue
        ## end of while

        classid+=1
    # end of 'for classid in range(numClasses)'
    return posSamples

def get_negative_samples(objData,object_id,min_samples,max_samples,overlap_ratio):

    numSamples = len(objData[object_id - 1])
    negSamples = []
    sampleid = 0
    sampleid_ = 0
    while sampleid < numSamples:
        action = objData[object_id - 1][sampleid]['action']
        if not action:
            sampleid_ = sampleid
            sequence = []
            count = 0
            for sampleid_ in range(sampleid, numSamples):
                action = objData[object_id - 1][sampleid_]['action']  # take the action name set
                if not action:   #  if action name set is empty
                    frame = objData[object_id - 1][sampleid_]['frame']
                    box = objData[object_id - 1][sampleid_]['box']
                    pose = objData[object_id - 1][sampleid_]['pose']
                    action = '-1'
                    if count < max_samples:  # 120
                        sequence.append({'frame': frame, 'object_id': object_id, 'box': box, 'action': action, 'pose': pose})
                    else: # if count >= max_samples,
                        ## if the max. amount of negative samples is reached, this sample will not be added to the sequence,
                        # start over again from the this sampleid in while
                        break
                else:
                    break
                sampleid_ += 1
                count += 1
            # end of 'for'

            if count >= min_samples:
                negSamples.append(sequence)
            sampleid = sampleid_
        else:
            sampleid += 1
            continue
    # end of while

    return negSamples

def extract_pose(data,object_id,PATH,prefix,poseData): #data are pos_sample or neg_samples : sequences of GT bounding boxes with label,    poseData are sequences of poses and pose bounding boxes
    #data[object_id - 1][all pos segment][specific element in segment]
    #print(len(data[object_id-1]))
    #print(data[object_id-1][0][0]['pose'].LElbow_X)
    overlap_ratio_thresh = 0.3
    dist_thresh = 40
    for sequenceid in range(len(data[object_id-1])):
        print('################## objectid:', object_id, ' sequenceid:', sequenceid)
        numFrames = len(data[object_id-1][sequenceid])
        action = data[object_id - 1][sequenceid][0]['action']  # action name of the first frame
        #print(frames)
        pose_world = np.array(np.zeros([2,18,numFrames]))
        count_missing = 0
        for framecount in range(numFrames):
            frameNr = data[object_id - 1][sequenceid][framecount]['frame']
            box = data[object_id - 1][sequenceid][framecount]['box']              ########### GT bounding box
            box_x = float(box.X)
            box_y = float(box.Y)
            box_w = float(box.Width)
            box_h = float(box.Height)
            box_x_left = box_x - box_w/2
            box_y_top = box_y - box_h/2
            # print(frame)
            numPerson = len(poseData[int(frameNr)])
            #print(box_x , box_y, numPerson, frame)
            dist = np.ones(numPerson)*100
            if not poseData[int(frameNr)]:   ################################################ if the pose frame is missing, fill with default -1
                pose_world[:, :, framecount] = -1
            else:   ##### if the pose frame is not missing
                ratioArea = np.zeros(numPerson)
                for posePersonID in range(0,numPerson):
                    poseBox_x = float(poseData[int(frameNr)][posePersonID]['box'].X)               ################ estimated bounding box based on pose
                    poseBox_y = float(poseData[int(frameNr)][posePersonID]['box'].Y)
                    poseBox_w = float(poseData[int(frameNr)][posePersonID]['box'].Width)
                    poseBox_h = float(poseData[int(frameNr)][posePersonID]['box'].Height)
                    poseBox_x_left = poseBox_x-poseBox_w/2
                    poseBox_y_top = poseBox_y-poseBox_h/2
                    # areaUnion = union_area((box_x_left,box_y_top,box_w,box_h),(poseBox_x_left,poseBox_y_top,poseBox_w,poseBox_h))
                    areaIntersection = intersection_area((box_x_left,box_y_top,box_w,box_h),(poseBox_x_left,poseBox_y_top,poseBox_w,poseBox_h))
                    ratioArea[posePersonID] = areaIntersection/(((box_w*box_h)+(poseBox_w*poseBox_h))-areaIntersection) # IoU
                    dist[posePersonID] = math.sqrt((int(box_x)-int(poseBox_x))**2+(int(box_y)-int(poseBox_y))**2)
                    #print(object_id, frame, box_x, box_y, poseBox_x, poseBox_y, areaUnion, areaIntersection, ratioArea[k], 'dist=', dist[k])
                posePersonWithSmallestDist = np.argmin(dist)
                maxNum = 0
                count = 0
                #########################################################################################################################################################
                for posePersonID in range(0,len(ratioArea)):    #  find out posePersonWithLargestIoU whose IoU is larger than 0.6; if no IoU > 0.6, then count == 0     # optimization!!!!!
                    if (ratioArea[posePersonID] > overlap_ratio_thresh and ratioArea[posePersonID] > maxNum):                                                                            #
                        maxNum = ratioArea[posePersonID]                                                                                                                #
                        posePersonWithLargestIoU = posePersonID                                                                                                         #
                        count +=1                                                                                                                                       #
                #########################################################################################################################################################
                if count == 0 :   # no posePersonWithLargestIoU larger than 0.6  is found
                    mappedPosePerson=-1
                    print('object_id: ', object_id, 'frameNr: ', frameNr, ' max overlapratio: ',
                          max(ratioArea), 'personNr: ', np.argmax(ratioArea), ' min dist:', dist[posePersonWithSmallestDist], 'personNr: ', posePersonWithSmallestDist, ' Overlap ratio too small. No Pose selected')
                else:
                    if dist[posePersonWithSmallestDist] < dist_thresh and posePersonWithLargestIoU==posePersonWithSmallestDist:
                        mappedPosePerson=posePersonWithLargestIoU
                    else:
                        mappedPosePerson=-1
                        print('object_id: ', object_id, 'frameNr: ', frameNr, ' max overlapratio: ',
                              ratioArea[posePersonWithLargestIoU], 'personNr: ', posePersonWithLargestIoU,' min dist:', dist[posePersonWithSmallestDist],'personNr: ', posePersonWithSmallestDist,
                              ' Min dist too big or not the person of largest iou. No Pose selected')

                if mappedPosePerson >= 0:
                    pose = poseData[int(frameNr)][mappedPosePerson]['pose']
                    print('object_id: ', object_id, 'frameNr: ', frameNr, ' max overlapratio: ',
                          ratioArea[posePersonWithLargestIoU], ' min dist:', dist[posePersonWithSmallestDist],
                          ' Pose Chosen!')

                    pose_world[0][0][framecount] = pose.LAnkle_X
                    pose_world[1][0][framecount] = pose.LAnkle_Y
                    pose_world[0][1][framecount] = pose.LEar_X
                    pose_world[1][1][framecount] = pose.LEar_Y
                    pose_world[0][2][framecount] = pose.LElbow_X
                    pose_world[1][2][framecount] = pose.LElbow_Y
                    pose_world[0][3][framecount] = pose.LEye_X
                    pose_world[1][3][framecount] = pose.LEye_Y
                    pose_world[0][4][framecount] = pose.LHip_X
                    pose_world[1][4][framecount] = pose.LHip_Y
                    pose_world[0][5][framecount] = pose.LKnee_X
                    pose_world[1][5][framecount] = pose.LKnee_Y
                    pose_world[0][6][framecount] = pose.LShoulder_X
                    pose_world[1][6][framecount] = pose.LShoulder_Y
                    pose_world[0][7][framecount] = pose.LWrist_X
                    pose_world[1][7][framecount] = pose.LWrist_Y
                    pose_world[0][8][framecount] = pose.Neck_X
                    pose_world[1][8][framecount] = pose.Neck_Y
                    pose_world[0][9][framecount] = pose.Nose_X
                    pose_world[1][9][framecount] = pose.Nose_Y
                    pose_world[0][10][framecount] = pose.RAnkle_X
                    pose_world[1][10][framecount] = pose.RAnkle_Y
                    pose_world[0][11][framecount] = pose.REar_X
                    pose_world[1][11][framecount] = pose.REar_Y
                    pose_world[0][12][framecount] = pose.RElbow_X
                    pose_world[1][12][framecount] = pose.RElbow_Y
                    pose_world[0][13][framecount] = pose.REye_X
                    pose_world[1][13][framecount] = pose.REye_Y
                    pose_world[0][14][framecount] = pose.RHip_X
                    pose_world[1][14][framecount] = pose.RHip_Y
                    pose_world[0][15][framecount] = pose.RKnee_X
                    pose_world[1][15][framecount] = pose.RKnee_Y
                    pose_world[0][16][framecount] = pose.RShoulder_X
                    pose_world[1][16][framecount] = pose.RShoulder_Y
                    pose_world[0][17][framecount] = pose.RWrist_X
                    pose_world[1][17][framecount] = pose.RWrist_Y
                else:   # mappedPosePerson = -1:
                    pose_world[0][0][framecount] = -1
                    pose_world[1][0][framecount] = -1
                    pose_world[0][1][framecount] = -1
                    pose_world[1][1][framecount] = -1
                    pose_world[0][2][framecount] = -1
                    pose_world[1][2][framecount] = -1
                    pose_world[0][3][framecount] = -1
                    pose_world[1][3][framecount] = -1
                    pose_world[0][4][framecount] = -1
                    pose_world[1][4][framecount] = -1
                    pose_world[0][5][framecount] = -1
                    pose_world[1][5][framecount] = -1
                    pose_world[0][6][framecount] = -1
                    pose_world[1][6][framecount] = -1
                    pose_world[0][7][framecount] = -1
                    pose_world[1][7][framecount] = -1
                    pose_world[0][8][framecount] = -1
                    pose_world[1][8][framecount] = -1
                    pose_world[0][9][framecount] = -1
                    pose_world[1][9][framecount] = -1
                    pose_world[0][10][framecount] = -1
                    pose_world[1][10][framecount] = -1
                    pose_world[0][11][framecount] = -1
                    pose_world[1][11][framecount] = -1
                    pose_world[0][12][framecount] = -1
                    pose_world[1][12][framecount] = -1
                    pose_world[0][13][framecount] = -1
                    pose_world[1][13][framecount] = -1
                    pose_world[0][14][framecount] = -1
                    pose_world[1][14][framecount] = -1
                    pose_world[0][15][framecount] = -1
                    pose_world[1][15][framecount] = -1
                    pose_world[0][16][framecount] = -1
                    pose_world[1][16][framecount] = -1
                    pose_world[0][17][framecount] = -1
                    pose_world[1][17][framecount] = -1
                    count_missing+=1
                    #print('No Pose Selected')
            framecount+=1
        print('MissingPose',count_missing, 'TotalFrames', numFrames)
        #print(pose_world)
        #print(pose_world.shape)
        if float(count_missing)/float(numFrames) < 0.5:
            pose_path = PATH + prefix +'_' 'id_' + str(object_id) + '_' + str(sequenceid+1) + '_' + action +'/'
            if not os.path.exists(pose_path):
                os.makedirs(pose_path)
            name = pose_path + 'joint_positions.mat'
            sio.savemat(name, mdict={'pose_world': pose_world})
            print('################## object_id: ', object_id,  'sequenceid:', sequenceid, ' numFrames:', numFrames)
        else:
            print('################## object_id: ', object_id ,'sequenceid:', sequenceid, ' numFrames:', numFrames, ' Ignored due to too many missing frames!')

        # sequenceid+=1
    return 0

def extract_image(data,object_id,PATH_store,PATH_load,prefix,img_prefix,flow_x_prefix,flow_y_prefix,org_width,org_height,extra_width,extra_height,aspect_ratio,output_height, numDigitsFrameNr):
    #  data are pos_sample or neg_samples : sequences of bounding boxes with label
    #print(len(data[object_id - 1]))
    extra_width = extra_width / 100.0    # in percentage
    extra_height = extra_height / 100.0
    sequenceid = 0
    for sequenceid in range(len(data[object_id - 1])):
        numFrames = len(data[object_id - 1][sequenceid])
        action = data[object_id - 1][sequenceid][0]['action']
        write_img_path = PATH_store + prefix + '_' 'id_' + str(object_id) + '_' + str(sequenceid + 1) + '_' + action + '/'
        if not os.path.exists(write_img_path):
            os.makedirs(write_img_path)
        #print(frames)
        for frameid in range(numFrames):
            frame = data[object_id - 1][sequenceid][frameid]['frame']
            box = data[object_id-1][sequenceid][frameid]['box']   #  the box of GT cvml
            box_x = int(box.X)
            box_y = int(box.Y)
            box_width = int(box.Width)
            box_height = int(box.Height)
            box_width=math.ceil((box_width*(1+extra_width)))
            box_height = math.ceil((box_height*(1+extra_height)))  # extra_width is in percentage
            if aspect_ratio != -1:
                if (box_width/box_height>= aspect_ratio):
                    box_height = math.ceil(box_width/aspect_ratio)  #  box_width/box_height >= aspect_ratio, adapt height to width
                else:
                    box_width = math.ceil(box_height*aspect_ratio)  #  box_width/box_height < aspect_ratio, adapt width to height

            if ((box_x-box_width/2.0) >= 0 and (box_x+box_width/2.0) <= org_width and (box_y-box_height/2.0) >= 0 and (box_y+box_height/2.0) <= org_height): # org_width: width of the entire image
                rect_x = round(box_x-box_width/2.0,0)
                rect_y = round(box_y-box_height/2.0,0)
                print(object_id,frame, rect_x, rect_y, box_width, box_height)
            elif((box_x-box_width/2.0) < 0 and (box_x+box_width/2.0) <= org_width and (box_y-box_height/2.0) >= 0 and (box_y+box_height/2.0) <= org_height):
                rect_x = 0.0
                rect_y = round(box_y-box_height/2.0,0)
                print(object_id,frame, rect_x, rect_y, box_width, box_height)
            elif((box_x-box_width/2.0) >= 0 and (box_x+box_width/2.0) <= org_width and (box_y-box_height/2.0) < 0 and (box_y+box_height/2.0) <= org_height):
                rect_x = round(box_x-box_width/2,0)
                rect_y = 0.0
                print(object_id,frame, rect_x, rect_y, box_width, box_height)
            else:
                rect_x = 0.0
                rect_y = 0.0
                print(object_id, frame, rect_x, rect_y, box_width, box_height)
                 ############################################################################################################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #img_path = '{}/{}{:08d}.png'.format(PATH_load,img_prefix,int(frame))                                                              #  the frame number starts from 0
            img_path = ('{}/{}{:0'+str(numDigitsFrameNr)+'d}.png').format(PATH_load, img_prefix, int(frame)  )  ################################  here should be int(frame) instead of int(frame)+1
            img = cv.imread(img_path)
            crop_img = img[int(rect_y):int(rect_y)+int(box_height),int(rect_x):int(rect_x)+int(box_width)]     ## error: "NoneType" object is not subscriptable
            write_img_name = ('{}/img_{:0'+str(numDigitsFrameNr)+'d}.png').format(write_img_path, frameid + 1)
            if output_height != -1:
                output_width = int(round(output_height*aspect_ratio,0))
                resize_img =cv.resize(crop_img,(output_width,output_height))
                cv.imwrite(write_img_name, resize_img)
            else:
                cv.imwrite(write_img_name,crop_img)

            if flow_x_prefix != '':   #  '' means there are no flow data
                flow_path = ('{}/{}{:0'+str(numDigitsFrameNr)+'d}.png').format(PATH_load, flow_x_prefix, int(frame))
                img = cv.imread(flow_path)
                crop_img = img[int(rect_y):int(rect_y) + int(box_height), int(rect_x):int(rect_x) + int(box_width)]
                write_img_name = ('{}/flow_x_{:0'+str(numDigitsFrameNr)+'d}.png').format(write_img_path, frameid + 1)
                if output_height != -1:
                    output_width = int(round(output_height * aspect_ratio, 0))
                    resize_img = cv.resize(crop_img, (output_width, output_height))
                    cv.imwrite(write_img_name, resize_img)
                else:
                    cv.imwrite(write_img_name, crop_img)

            if flow_y_prefix != '':  #  '' means there are no flow data
                flow_path = ('{}/{}{:0'+str(numDigitsFrameNr)+'d}.png').format(PATH_load, flow_y_prefix, int(frame))
                img = cv.imread(flow_path)
                crop_img = img[int(rect_y):int(rect_y) + int(box_height), int(rect_x):int(rect_x) + int(box_width)]
                write_img_name = ('{}/flow_y_{:0'+str(numDigitsFrameNr)+'d}.png').format(write_img_path, frameid + 1)
                if output_height != -1:
                    output_width = int(round(output_height * aspect_ratio, 0))
                    resize_img = cv.resize(crop_img, (output_width, output_height))
                    cv.imwrite(write_img_name, resize_img)
                else:
                    cv.imwrite(write_img_name, crop_img)

            frameid+=1

    return 0

if __name__ == '__main__': # this code block can be executed only when you want to run this module as a program but not when someone wants to import this module and call the function themselves
    # CVML_file = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/RawData/GT/Action/RecordingBangalore/SAM_5734_TestRecording_B_HD.cvml'
    # CVML_cmuPose_file = \
    #     '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Intermediate_results/CMU_Pose_Airport/cvml/SAM_5734_testRecording_B_HD/SAM_5734_testRecording_B_HD_cmuPose.cvml'
    # path_pose = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport/Pose_SAM_5734_Lin'
    # path_img = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport/RGB_SAM_5734_Lin'
    # load_img = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/RawData/RecordingBangalore/png_start_0/SAM_5734_TestRecording_B_HD'
    # file_prefix = 'SAM_5734_TestRecording_B_HD'
    # img_prefix = 'SAM_5734-'

    CVML_file = '/home/lin7lr/ActionRecognitionGitLabCopy/data/GroundTruth/RecordingBangalore/Lobby_Laptop6_S1_15_Jan.cvml'
    CVML_cmuPose_file = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Intermediate_results/CMU_Pose_Airport/cvml/Laptop6_S1/Laptop6_S1.cvml'
    path_pose_pos = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/Activity/Pose_Laptop6_S1/Pos/'
    path_pose_neg = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/Activity/Pose_Laptop6_S1/Neg/'
    # path_img_pos = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/Activity/RGB_Laptop1/Pos/'
    # path_img_neg = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/Activity/RGB_Laptop1/Neg/'
    # load_img = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/RawData/RecordingBangalore/png_start_0/20170517_Lobby_Laptop1_S2' ################# no / at the end
    file_prefix = 'Laptop6_S1-'
    # img_prefix = 'fc2_save_2017-05-17-163808-'
    # numDigitsFrameNr = 4



    # flow_x_prefix = 'flow_x_'
    # flow_y_prefix = 'flow_y_'
    flow_x_prefix = ''
    flow_y_prefix = ''
    # classes = ['PickUp_comp', 'PickUp_shrink', 'PutBack_comp', 'PutBack_stretch', 'PickOut', 'PutInBag' ]
    classes = ['GetCheckedMD', 'PutLuggAtDesk', 'ScanPassengerMD', 'TakeLuggFrDesk']
    extra_width = 0 #in percentage 1 means 1 percent
    extra_height = 0 #in percentage 1 means 1 percent
    #aspect_ratio = round(340.0/256.0,3) #width/height
    aspect_ratio = -1
    output_height = -1 # -1 for default bounding box height
    maxFrame = 60580000
    minPosSegment = 10
    maxPosSegment = math.inf
    minNegSegment = 10
    # maxNegSegment = 120
    maxNegSegment = math.inf
    overlap_ratio = 0  #  not used
    numPose = 18 # Parameter not used yet
    print('Parsing CVML File........')   ################## GT cvml file
    tree = ET.parse(CVML_file)
    root = tree.getroot()

    print('Parsing CVML_Pose File........')  ###################### pose cvml file
    treePose = ET.parse(CVML_cmuPose_file)
    rootPose = treePose.getroot()

    for resolution in root.iter('resolution'):  # resolution in GT cvml file defines the orginal width and height
        org_width = int(resolution.get('width'))
        org_height = int(resolution.get('height'))

    object_id = []
    for object in root.iter('object'):
        object_id.append(int(object.get('id')))
    numObject = np.amax(object_id)
    print(numObject)

    frame = []
    for object in rootPose.iter('frame'):
        frame.append(int(object.get('number')))
    numFrame = np.amax(frame)
    #print(numFrame)

    if maxFrame <= numFrame:
        numFrame = maxFrame

###########Get Object Data########### #################################   GT cvml file
    objData = []
    for objectid in range(1,numObject+1):
        objData.append(get_object_data(objectid,root,maxFrame))
    #print(objData)

    poseData = []    ######################################### from Pose cvml file
    for frameid in range(0, numFrame+1):
        poseData.append(get_pose_data(frameid,rootPose))
    #print(poseData[1][0]['box'].X,poseData[1][0]['box'].Y)



# ##########Get Positive and Negative Samples###############
    pos_samples = []
    for objectid in range(1, numObject+1):
        pos_samples.append(get_positive_samples(objData,objectid,classes,minPosSegment,maxPosSegment))
    #print(len(pos_samples))
    #print(pos_samples[0])

    neg_samples = []
    for objectid in range(1, numObject+1):
        neg_samples.append(get_negative_samples(objData,objectid,minNegSegment,maxNegSegment,overlap_ratio))
    # print(neg_samples)
    #print(neg_samples[0])

########Extract Pose Data############

    for objectid in range(1, numObject+1):
        print('############################### positive pose samples of object_id {}:'.format(objectid))
        extract_pose(pos_samples,objectid,path_pose_pos,file_prefix,poseData)
        print('############################### negative pose samples of object_id {}:'.format(objectid))
        extract_pose(neg_samples, objectid, path_pose_neg, file_prefix,poseData)

# ##################### Extract Image Sequences ############################
#     for objectid in range(1, numObject+1):
#         extract_image(pos_samples, objectid, path_img_pos, load_img, file_prefix, img_prefix, flow_x_prefix, flow_y_prefix,
#                   org_width, org_height, extra_width, extra_height, aspect_ratio, output_height, numDigitsFrameNr)
#         extract_image(neg_samples, objectid, path_img_neg, load_img, file_prefix, img_prefix, flow_x_prefix, flow_y_prefix,
#                       org_width, org_height, extra_width, extra_height, aspect_ratio, output_height, numDigitsFrameNr)




