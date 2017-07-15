import tensorflow as tf
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

image_dir = 'images'
batch_size = 100

image_list = list(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg'))


writer = tf.python_io.TFRecordWriter('./train.tfrecords')
for s in image_list[0:50]:
    try:
        img = cv2.imread(s)
        img = cv2.resize(img, (512, 512))
        img_raw = img.tobytes()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            ))
        writer.write(example.SerializeToString())
    except:
        pass
writer.close()


def read_and_decode(file_list):
    '''
    given the tfrecord file list, construct a queue to produce data
    :param file_list: 
    :return: a tensor of train or test data
    '''
    file_queue = tf.train.string_input_producer(file_list, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    img = tf.decode_raw(features['img_raw'], tf.uint8)  # because it is between [0, 255]
    img = tf.reshape(img, [512, 512, 3])
    # img = tf.cast(img, tf.float32) / 255 - 0.5
    return img


with tf.Session() as sess:
    img = read_and_decode(['train.tfrecords'])
    print(img.shape)

    img_batch = tf.train.shuffle_batch([img], num_threads=1, batch_size=10, capacity=1000, min_after_dequeue=10)
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners(sess)
    a = sess.run(img_batch)
    print(a.shape)
    plt.imshow(a[0])
    plt.show()
    time.sleep(100)