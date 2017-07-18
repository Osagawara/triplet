import tensorflow as tf
import numpy as np
import triplet
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

log_dir = './logs'
tfrecord_dir = 'tfrecord/'
tfrecord_list = list(os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord'))
batch_size = 20
global_step = 0

def read_and_decode(file_list):
    '''
    given the tfrecord file list, construct a queue to produce data
    :param file_list: 
    :return: a tensor with shape [3, 512, 512, 3]
    '''
    file_queue = tf.train.string_input_producer(file_list, num_epochs=20)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'triplet': tf.FixedLenFeature([3], tf.string)
        }
    )

    img = tf.decode_raw(features['triplet'], tf.float32)  # because it is between [0, 255]
    # img = tf.cast(img, tf.float32) / 255 - 0.5
    a = tf.reshape(img[0],(1, 512 ,512 ,3))
    p = tf.reshape(img[1],(1, 512 ,512 ,3))
    n = tf.reshape(img[2],(1, 512 ,512 ,3))

    triplet = tf.concat((a, p, n), axis=0)
    return triplet

with tf.Session() as sess:
    triplet_single = read_and_decode(tfrecord_list)
    triplet_batch = tf.train.shuffle_batch([triplet_single], num_threads=4, batch_size=batch_size, capacity=1000,
                                       min_after_dequeue=100)

    # the shape of the triplet_batch is [batch_size, 3, 512, 512, 3]
    # the shape of anchor is [batch_size, 512, 512, 3]
    anchor = triplet_batch[:, 0, :, :, :]
    positive = triplet_batch[:, 1, :, :, :]
    negative = triplet_batch[:, 2, :, :, :]

    # big_batch is the raw triplet to be selected
    # data_holder is the placeholder where we will put the training data in it
    big_batch = tf.concat((anchor, positive, negative), axis=0)
    data_holder = tf.placeholder(tf.float32, (3*batch_size, 512, 512, 3), name='train_data')

    print(big_batch.shape)
    triple = triplet.Triplet()
    triple.build(data_holder)

    tf.summary.scalar(name='loss', tensor=triple.hard_loss)
    learning_rate = tf.train.exponential_decay(0.0001, 20000, 200, 0.98)
    tf.summary.scalar(name='learning rate', tensor=learning_rate)
    train = tf.train.AdamOptimizer(learning_rate).minimize(triple.hard_loss)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess)


    # while(global_step < 20001):
        # prepare the data
    a_list = []
    p_list = []
    n_list = []
    while len(a_list) < 20:

        print("aaa")

        t = sess.run(big_batch)
        print("bbb")
        s = time.time()
        semi_loss = sess.run(triple.semi_loss, feed_dict={data_holder:t})
        print("time is {}".format(time.time() - s))
        index = semi_loss < 0
        print("sample number {}".format(np.sum(index)))
        a_list += np.split(t[0:batch_size][index], np.sum(index))
        p_list += np.split(t[batch_size:2 * batch_size][index], np.sum(index))
        n_list += np.split(t[2 * batch_size: 3 * batch_size][index], np.sum(index))

    t = a_list[0:batch_size] + p_list[0:batch_size] + n_list[0:batch_size]
    t = list(f.reshape((1, 512, 512, 3)) for f in t)
    t = np.concatenate(t, axis=0)

    print(t.shape)

        # thre train step
    _, summary = sess.run([train, merged], feed_dict={data_holder: t})
    if global_step % 10 == 0:
        train_writer.add_summary(summary, global_step)
    if global_step % 50 == 0:
        print("Global Step {} OK".format(global_step))
    if global_step % 1000 == 0:
        triple.save_npy(sess=sess, npy_path='triplet_lr_1e-4.npy')
    global_step += 1
        
    train_writer.close()