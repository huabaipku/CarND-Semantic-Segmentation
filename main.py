import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    ## ============ step 1: load vgg ============ ##
    ## get input, keep_prob, and 3 layers (3,4,7) ##
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    with tf.name_scope('From_VGG'):
        x = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return x, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    ## =============== step 2: create FCN decoder layers =============== ##
    ## First connect the vgg layer (3, 4, 7) to a 1x1 convolution layer, ##
    ## in which the last dimention will be num_classes, in this case 2   ##
    ## Then, combine with previous layers

    ## connect Layer 7 to 1x1 conv layer, and then decoding
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name="layer7_conv_1x1")
    output = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, strides=(2, 2), padding="same",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        name="layer7_decoded")

    ## connect Layer 4 to 1x1 conv layer, add decoded layer7, and then decoding again
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name="layer4_conv_1x1")
    output = tf.add(output, conv_1x1_layer4)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2), padding="same",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        name="layer4plus7_decoded")

    ## connect Layer 3 to 1x1 conv layer, add decoded (layer4 + layer7), and then decoding again
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name="layer3_conv_1x1")
    output = tf.add(output, conv_1x1_layer3)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8), padding="same",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        name="layer3plus4plus7__decoded")
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    ## =============== step 3: add AdamOptimizer =============== ##
    ## First, flatten the last output layer and the labels       ##
    ## Then, define the lose function                            ##
    ## Finally, choose tf.train.AdamOptimizer                    ##

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    with tf.name_scope('cross_entropy'):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = learning_rate
    lr = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)
    lr = tf.Print(lr, [lr], "Current Learning Rate is:")
    tf.summary.scalar("learning_rate", lr)

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy_loss, global_step=global_step)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, summary_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    ## =============== step 4: set up the training process =============== ##
    ##        ##
    print("========= Start training process =============")

    logs_path = './logs'
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    overall_index = 1
    for epoch in range(epochs):
       print("=========== EPOCH #: {} =============".format(epoch+1))
       batch_index = 1
       for batch_images, batch_labels in get_batches_fn(batch_size):
           param_dict = {input_image: batch_images,
                         correct_label: batch_labels,
                         keep_prob: 0.5,
                         learning_rate: 0.0003}
           _,summary, loss = sess.run([train_op, summary_op, cross_entropy_loss], feed_dict=param_dict)
           if batch_index%10 == 0:
               print("==========loss:{}========".format(loss))
           writer.add_summary(summary, overall_index)
           batch_index += 1
           overall_index += 1
    # pass
# tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    EPOCHS = 50
    BATCH_SIZE = 8

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        print("===== Builed_NN ======")

        with tf.name_scope("final_layer"):
            final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        print("===== Get final layer =====")

        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, cross_entropy_loss  = optimize(final_layer, correct_label, learning_rate, num_classes)
        tf.summary.scalar("cost", cross_entropy_loss)
        summary_op = tf.summary.merge_all()

        print("===== set up optimizer")

        # TODO: Train NN using the train_nn function

        sess.run(tf.global_variables_initializer())
        print("==== Initialize the global variables")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, summary_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
