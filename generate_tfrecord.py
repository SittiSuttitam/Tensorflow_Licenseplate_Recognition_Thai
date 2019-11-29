"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'licenseplate':
        return 1
    elif row_label == 'gorgai':
        return 2
    elif row_label == 'korkwaai':
        return 3
    elif row_label == 'korrakung':
        return 4
    elif row_label == 'ngorngoo':
        return 5
    elif row_label == 'jorjaan':
        return 6
    elif row_label == 'chorching':
        return 7
    elif row_label == 'chorchang':
        return 8
    elif row_label == 'sorsoh':
        return 9
    elif row_label == 'chorcher':
        return 10
    elif row_label == 'yorying':
        return 11
    elif row_label == 'dorchada':
        return 12
    elif row_label == 'tortaan':
        return 13
    elif row_label == 'nornayn':
        return 14
    elif row_label == 'dordek':
        return 15
    elif row_label == 'tortao':
        return 16
    elif row_label == 'tortung':
        return 17
    elif row_label == 'tortahaan':
        return 18
    elif row_label == 'tortong':
        return 19
    elif row_label == 'nornoo':
        return 20
    elif row_label == 'borbaimai':
        return 21
    elif row_label == 'porpla':
        return 22
    elif row_label == 'porping':
        return 23
    elif row_label == 'porpaan':
        return 24
    elif row_label == 'forfun':
        return 25
    elif row_label == 'porsumpao':
        return 26
    elif row_label == 'mormaa':
        return 27
    elif row_label == 'yoryuk':
        return 28
    elif row_label == 'roreua':
        return 29
    elif row_label == 'lorling':
        return 30
    elif row_label == 'worwaen':
        return 31
    elif row_label == 'sorseua':
        return 32
    elif row_label == 'horheep':
        return 33
    elif row_label == 'lorjula':
        return 34
    elif row_label == 'oraang':
        return 35
    elif row_label == 'hornokhook':
        return 36
    elif row_label == 'korkai':
        return 37
    elif row_label == 'korkuat':
        return 38
    elif row_label == 'korkon':
        return 39
    elif row_label == 'dtorpadtak':
        return 40
    elif row_label == 'tormontoh':
        return 41
    elif row_label == 'torpootao':
        return 42
    elif row_label == 'forfaa':
        return 43
    elif row_label == 'sorsala':
        return 44
    elif row_label == 'soreusee':
        return 45
    elif row_label == 'one':
        return 46
    elif row_label == 'two':
        return 47
    elif row_label == 'three':
        return 48
    elif row_label == 'four':
        return 49
    elif row_label == 'five':
        return 50
    elif row_label == 'six':
        return 51
    elif row_label == 'seven':
        return 52
    elif row_label == 'eight':
        return 53
    elif row_label == 'nine':
        return 54
    elif row_label == 'zero':
        return 55
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
