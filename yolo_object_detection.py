import cv2
import numpy as np
import glob
import random

# Load Yolo
net_crane = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
net_load = cv2.dnn.readNet("yolov3_training_load.weights", "yolov3_testing.cfg")
# net_person = cv2.dnn.readNet("yolov3_training_person.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["crane"]

# Images path
images_path = glob.glob(r"C:\Users\Hp\PycharmProjects\IIQ-project\test_images\*.jpg")

width = 640
height=480

layer_names_crane = net_crane.getLayerNames()
layer_names_load = net_load.getLayerNames()
# layer_names_person = net_person.getLayerNames()

output_layers_crane = []
output_layers_load = []
output_layers_person = []
for i in net_crane.getUnconnectedOutLayers():
    output_layers_crane.append(layer_names_crane[i-1])
    # print(i)
for i in net_load.getUnconnectedOutLayers():
    output_layers_load.append(layer_names_load[i-1])
    # print(i)
# for i in net_person.getUnconnectedOutLayers():
#     output_layers_person.append(layer_names_person[i-1])
#     # print(i)

# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors_crane = np.random.uniform(0, 255, size=(len(classes), 3))
colors_load = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    # img = cv2.resize(img, None, width= width,height = height)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net_load.setInput(blob)
    net_crane.setInput(blob)
    # net_person.setInput(blob)
    outs_crane = net_crane.forward(output_layers_crane)
    outs_load = net_load.forward(output_layers_load)
    # outs_person = net_load.forward(output_layers_person)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    items = []
    for out in outs_crane:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.59:
                # Object detected
                print(class_id)
                center_x_crane = int(detection[0] * width)
                center_y_crane = int(detection[1] * height)
                w_crane = int(detection[2] * width)
                h_crane = int(detection[3] * height)

                # Rectangle coordinates
                x_crane = int(center_x_crane - w_crane / 2)
                y_crane = int(center_y_crane - h_crane / 2)

                boxes.append([x_crane, y_crane, w_crane, h_crane])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                items.append(0)
    # for out in outs_person:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         if confidence > 0.59:
    #             # Object detected
    #             print(class_id)
    #             center_x_person = int(detection[0] * width)
    #             center_y_person = int(detection[1] * height)
    #             w_person = int(detection[2] * width)
    #             h_person = int(detection[3] * height)
    #
    #             # Rectangle coordinates
    #             x_person = int(center_x_person - w_person / 2)
    #             y_person = int(center_y_person - h_person / 2)
    #
    #             boxes.append([x_person, y_person, w_person, h_person])
    #             confidences.append(float(confidence))
    #             class_ids.append(class_id)
    #             items.append(2)
    for out in outs_load:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.59:
                # Object detected
                print(class_id)
                center_x_load = int(detection[0] * width)
                center_y_load = int(detection[1] * height)
                w_load = int(detection[2] * width)
                h_load = int(detection[3] * height)

                # Rectangle coordinates
                x_load = int(center_x_load - w_load / 2)
                y_load = int(center_y_load - h_load / 2)

                boxes.append([x_load, y_load, w_load, h_load])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                items.append(1)

    indexes = cv2.dnn.NMSBoxes(boxes[0:4], confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            # x, y, w, h, item = boxes[i]
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color_crane = colors_crane[class_ids[i]]
            color_load = colors_load[class_ids[i]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color_crane, 2)
            # cv2.putText(img, "Crane", (x, y), font, 0.5, color_load, 2)
            if items[i] == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), color_crane, 2)
                cv2.putText(img, "Crane", (x, y), font, 0.5, color_crane, 2)
            elif items[i] == 1:
                cv2.rectangle(img, (x, y), (x + w, y + h), color_load, 2)
                cv2.putText(img, "Load", (x, y), font, 0.5, color_load, 2)
            # elif items[i] == 2:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), color_person, 2)
            #     cv2.putText(img, "Load", (x, y), font, 0.5, color_person, 2)



    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()