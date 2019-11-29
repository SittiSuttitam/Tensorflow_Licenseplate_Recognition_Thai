# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import image_conversion as dic
import allow_needed_values as anv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'test1.jpg' 
IMAGE_NAME2 = 'image0.png'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def GetClassName(data):
    for cl in data:
        return cl['name']


PATH_TO_TEST_IMAGES_DIR = 'test_Image/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test{}.jpg'.format(i)) for i in range(5, 6) ]
IMAGE_SIZE = (12, 8)

PATH_TO_IMAGE2 = os.path.join(CWD_PATH,IMAGE_NAME2)

# Number of classes the object detector can identify
NUM_CLASSES = 55

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier


# Input tensor is the image

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
# Perform the actual detection by running the model with the image as input
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
                ymin = boxes[0, 0, 0]
                xmin = boxes[0, 0, 1]
                ymax = boxes[0, 0, 2]
                xmax = boxes[0, 0, 3]
                (im_width, im_height) = image.size
                (xminn, xmaxx, yminn, ymaxx) = (
                xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn), int(ymaxx - yminn),int(xmaxx - xminn))
                img_data = sess.run(cropped_image)

                gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                gray = cv2.medianBlur(gray, 3)
                filename = os.path.join(CWD_PATH, 'image0.png')
                cv2.imwrite(filename, gray)
                image2 = cv2.imread(PATH_TO_IMAGE2)
                image_expanded = np.expand_dims(image2, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes2, scores2, classes2, num2) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image2,
                    np.squeeze(boxes2),
                    np.squeeze(classes2).astype(np.int32),
                    np.squeeze(scores2),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.70)

                #vis_util.visualize_boxes_and_labels_on_image_array(
                 #   image_np,
                 #   np.squeeze(boxes),
                 #   np.squeeze(classes).astype(np.int32),
                 #   np.squeeze(scores),
                 #   category_index,
                 #   use_normalized_coordinates=True,
                  #  line_thickness=5)

                #countSum = 0
                #count = 0
                #index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                #for i in index:

                    ##cv2.imshow("Crop{}".format(i), img_data)
                    #count = 0
                    #filename, countSum = dic.yo_make_the_conversion(img_data, countSum)
        cv2.imshow("Crop Image", image2)
        min_score_thresh = 0.8
        #data = [category_index.get(value) for index,value in enumerate(classes2[0]) if scores2[0,index] > min_score_thresh]

        for index, value in enumerate(classes2[0]):
            object_dict = {}
            if scores2[0, index] > min_score_thresh:
                print(GetClassName([(category_index.get(value))]))
        print (boxes2)


# All the results have been drawn on image. Now display the image.

# Press any key to close the image
        cv2.waitKey(0)

# Clean up
        cv2.destroyAllWindows()

