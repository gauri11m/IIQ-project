import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Resources\TestVideo.mp4")
wht = 320
thresh = 0.2
nmsthresh = 0.2

classFile = 'coco.names'
classes = []
with open(classFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfig =  'yolov3_testing.cfg'
modelWeights_crane = 'yolov3_training_crane.weights'
modelWeights_person = 'yolov3_training_load.weights'
modelWeights_load = 'yolov3_training_load.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights_crane)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net_person = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights_person)
net_person.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_person.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net_load = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights_load)
net_load.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_load.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def find_objects(outputs,img):
    ht,wt,ct = img.shape
    box = []
    class_ids = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh:
                w,h = int(det[2]*wt), int(det[3]*ht)
                x,y = int(det[0]*wt - w/2), int(det[1]*ht - h/2)
                box.append([x,y,w,h])
                class_ids.append(class_id)
                confs.append(float(confidence))
    print(len(box))
    indices = cv2.dnn.NMSBoxes(box,confs,thresh,nmsthresh)

    for i in indices:
        # i = i[0]
        bbox = box[i]
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        # cv2.putText(img, f'{classes[class_ids[i]].upper()} {int(confs[i]*100)}%',
        #             (x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        cv2.putText(img, 'load'+ f'{int(confs[i]*100)}%' ,
                    (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    # img = cv2.resize(img)

    img = cv2.resize(img, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    # cut_image = img[426: 853, 0: 1280]

    blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),(0,0,0),1,crop=False)
    net.setInput(blob)
    net_person.setInput(blob)

    layerNames = net.getLayerNames()
    layerNames_person = net_person.getLayerNames()
    outputNames = []
    outputNames_person = []
    # print(layerNames)
    for i in net.getUnconnectedOutLayers():
        outputNames.append(layerNames[i - 1])
    for i in net_person.getUnconnectedOutLayers():
        outputNames_person.append(layerNames_person[i - 1])
    # outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    outputs_person = net.forward(outputNames_person)
    # print(outputs[0].shape)

    ht, wt, ct = img.shape
    box = []
    class_ids = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh:
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int(det[0] * wt - w / 2), int(det[1] * ht - h / 2)
                box.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))
    print(len(box))
    indices = cv2.dnn.NMSBoxes(box, confs, thresh, nmsthresh)

    for i in indices:
        # i = i[0]
        bbox = box[i]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cut_image = img[y-10: y+h+100, x-10: x+w+10]
        cv2.imshow('Image_crop', cut_image)
        blob_load = cv2.dnn.blobFromImage(cut_image, 1 / 255, (wht, wht), (0, 0, 0), 1, crop=False)
        net_load.setInput(blob_load)
        layerNames_load = net_load.getLayerNames()
        outputNames_load = []
        # print(layerNames)
        for i in net_load.getUnconnectedOutLayers():
            outputNames_load.append(layerNames_load[i - 1])
        outputs_load = net_load.forward(outputNames_load)

        find_objects(outputs_load, cut_image)
        # cv2.putText(img, f'{classes[class_ids[i]].upper()} {int(confs[i]*100)}%',
        #             (x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        cv2.putText(img, 'crane',
                    (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow('Image',img)

    cv2.waitKey(1)