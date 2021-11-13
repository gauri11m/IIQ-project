import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import albumentations as A
import cv2
import os
import random
from PIL import Image
random.seed(7)

images_path = r"C:\Users\Hp\PycharmProjects\\45degreerule_final\\resources\processing_cranes\images"
annotations_path = r"C:\Users\Hp\PycharmProjects\\45degreerule_final\\resources\processing_cranes\\annotations"
new_images_path = r"C:\Users\Hp\PycharmProjects\\45degreerule_final\\resources\processing_cranes\\new_images"
new_annotations_path = r"C:\Users\Hp\PycharmProjects\\45degreerule_final\\resources\processing_cranes\\new_annotations"
# image = cv2.imread("resources\\test_images\TestImage107.jpg")

# width, height = 640, 480
# image = cv2.resize(image, (width, height))
# image = cv2.medianBlur(image, 5)
# image = cv2.flip(image,1)


def visualize(image):
  plt.figure(figsize=(10, 10))
  plt.axis('off')
  plt.imshow(image)
  plt.show()


def plot_example(image, bboxes):
  fig = plt.figure(figsize=(15, 15))
  if bboxes is not None:
    for bbox in bboxes:
      x_min = int(bbox[0])
      y_min = int(bbox[1])
      x_max = int(bbox[2])
      y_max = int(bbox[3])
      cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
    plt.imshow(image)
    plt.show()


def plotExamples(images, all_bboxes):
  i = 0
  for bboxes in all_bboxes:
    plot_example(images[i], bboxes)
    i += 1

def readImage(filename):
  # Reading Image using PIL
  # images_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/Fire_Detection-2/Images'
  im_path = images_path + '/' + filename
  os.chdir(images_path)
  image = Image.open(filename)
  image = np.array(image)
  return image


def getCoordinates(filename):
  # annotations_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/Fire_Detection-2/XML'
  os.chdir(annotations_path)
  my_tree = ET.parse(filename)
  my_root = my_tree.getroot()

  #  getting the coordinates
  allbb = []
  for obj in my_root.findall('object'):
    classid = str(obj.find('name').text)
    bndbox = obj.find('bndbox')
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    b = [xmin, ymin, xmax, ymax, classid]
    allbb.append(b)
  # print(allbb)
  return allbb


#  modidy the XML with new coordinates and store in a new place
def modifyXML_andSave(filename, new_image_name, bboxes):
  # annotations_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/Fire_Detection-2/XML'
  # new_annotations_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/new_anns'
  os.chdir(annotations_path)
  my_tree = ET.parse(filename)
  my_root = my_tree.getroot()
  my_root.find('filename').text = new_image_name
  for bbox in bboxes:
    name = bbox[-1]
    # print(name)
    for obj in my_root.findall('object'):
      obj_name = obj.find('name')
      if (obj_name.text == name):
        # print("right object found")
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(bbox[0])
        bndbox.find('xmax').text = str(bbox[1])
        bndbox.find('ymin').text = str(bbox[2])
        bndbox.find('ymax').text = str(bbox[3])
        # bndbox.find('xmin').text = str("{:.2f}".format(bbox[0]))
        # bndbox.find('xmax').text = str("{:.2f}".format(bbox[1]))
        # bndbox.find('ymin').text = str("{:.2f}".format(bbox[2]))
        # bndbox.find('ymax').text = str("{:.2f}".format(bbox[3]))
  new_xml = new_image_name.split('.')[0] + '.xml'
  os.chdir(new_annotations_path)
  my_tree.write(new_xml)



def save_aug_image(filename, image):
  # new_images_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/new_ims'
  # print(image.dtype)
  im = Image.fromarray(image)
  os.chdir(new_images_path)
  im.save(filename)


def main_call():
  # defining the transforms
  #  TRANSFORMS
  ################################
  # ADD MORE TRANSFOPRMS IF NEEDED AS PER REQUIRED AUGMENTATIONS
  ################################
  transform = A.Compose([
    #  A.Resize(width = 1920,height=1080),
    A.HorizontalFlip(p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.OneOf([
      A.Blur(blur_limit=3, p=0.4),
      A.ColorJitter(brightness=0.3, contrast=0.2, p=0.4),
    ], p=0.4),
    A.Rotate(limit=30, p=0.4)
  ], bbox_params=A.BboxParams(format='pascal_voc'))

  # #############

  # images_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/Fire_Detection-2/Images'
  # annotations_path = '/Users/student/Work/IIQ/Augmentations_fire_kurle/Fire_Detection-2/XML'
  images = os.listdir(images_path)
  annotations = os.listdir(annotations_path)
  i = 0
  for image in images:
    i += 1
    if i == 2:
      print("done test")
      break
    img = readImage(image)
    ann_path = image.split('.')[0] + '.xml'
    annotation = annotations[annotations.index(ann_path)]
    bbox = getCoordinates(annotation)
    images_list = []
    saved_bbox = []
    #####################
    num_of_annotation = 30  # change as per need of augmentated images
    #####################
    for i in range(num_of_annotation):
      augmentations = transform(image=img, bboxes=bbox)
      augmented_img = augmentations['image']
      images_list.append(augmented_img)
      saved_bbox.append(augmentations['bboxes'])
      new_filename = image.split('.')[0] + '_' + str(i) + '.jpg'
      save_aug_image(new_filename, augmented_img)
      modifyXML_andSave(annotation, new_filename, augmentations['bboxes'])
      print("xml and image saved for image " + new_filename)
      # print(len(augmentations['bboxes']))
    # plotExamples(images_list,saved_bbox)


if __name__ == '__main__':
  # getCoordinates('/content/drive/MyDrive/Augmentations/annotations/hm_0.xml')
  main_call()
# sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#
# image = cv2.filter2D(image, -1, sharpen)
#
# cv2.imshow('final', image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

