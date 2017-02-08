import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os

CONTENT_DIR = './images'
STYLE_DIR = './styles'
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.jpg'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
INI_NOISE_RATIO = 0.7
STYLE_WEIGHT = 500
ITERATION = 250


CONTENT_LAYERS = [('conv4_2', 1.)]
# STYLE_LAYERS = [('conv1_1', 0.5), ('conv2_1', 1.), ('conv3_1', 1.5), ('conv4_1', 3.), ('conv5_1', 4.)]
# STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.5),('conv3_1',2.),('conv4_1',2.5),('conv5_1',3.)]
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def load_image(path):
    image = scipy.misc.imread(path)
    # shape (h, w, d) to (1, h, w, d)
    image = image[np.newaxis, :, :, :]
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image


def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # shape (1, h, w, d) to (h, w, d)
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def noise_image(content_img):
    _, h, w, d = content_img.shape
    init_img = np.random.uniform(-20, 20, (1, h, w, d)).astype('float32')
    init_img = INI_NOISE_RATIO * init_img + (1. - INI_NOISE_RATIO) * content_img
    return init_img


def build_vgg19(img):
    def conv_layer(layer_input, weight_bias):
        conv = tf.nn.conv2d(layer_input, weight_bias[0], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv + weight_bias[1])
        return relu

    def pool_layer(layer_input):
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    def get_weight_bias(i):
        weight = vgg_layers[i][0][0][0][0][0]
        weight = tf.constant(weight)
        bias = vgg_layers[i][0][0][0][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weight, bias

    vgg_rawnet = scipy.io.loadmat(VGG_MODEL)
    vgg_layers = vgg_rawnet['layers'][0]

    net = dict()
    _, h, w, d = img.shape
    net['input'] = tf.Variable(np.zeros((1, h, w, d)).astype('float32'))
    # LAYER GROUP 1
    net['conv1_1'] = conv_layer(net['input'], get_weight_bias(0))
    net['conv1_2'] = conv_layer(net['conv1_1'], get_weight_bias(2))
    net['pool1'] = pool_layer(net['conv1_2'])
    # LAYER GROUP 2
    net['conv2_1'] = conv_layer(net['pool1'], get_weight_bias(5))
    net['conv2_2'] = conv_layer(net['conv2_1'], get_weight_bias(7))
    net['pool2'] = pool_layer(net['conv2_2'])
    # LAYER GROUP 3
    net['conv3_1'] = conv_layer(net['pool2'], get_weight_bias(10))
    net['conv3_2'] = conv_layer(net['conv3_1'], get_weight_bias(12))
    net['conv3_3'] = conv_layer(net['conv3_2'], get_weight_bias(14))
    net['conv3_4'] = conv_layer(net['conv3_3'], get_weight_bias(16))
    net['pool3'] = pool_layer(net['conv3_4'])
    # LAYER GROUP 4
    net['conv4_1'] = conv_layer(net['pool3'], get_weight_bias(19))
    net['conv4_2'] = conv_layer(net['conv4_1'], get_weight_bias(21))
    net['conv4_3'] = conv_layer(net['conv4_2'], get_weight_bias(23))
    net['conv4_4'] = conv_layer(net['conv4_3'], get_weight_bias(25))
    net['pool4'] = pool_layer(net['conv4_4'])
    # LAYER GROUP 5
    net['conv5_1'] = conv_layer(net['pool4'], get_weight_bias(28))
    net['conv5_2'] = conv_layer(net['conv5_1'], get_weight_bias(30))
    net['conv5_3'] = conv_layer(net['conv5_2'], get_weight_bias(32))
    net['conv5_4'] = conv_layer(net['conv5_3'], get_weight_bias(34))
    net['pool5'] = pool_layer(net['conv5_4'])
    return net


def content_layer_loss(p, x):
    _, h, w, d = p.shape
    M = h * w
    N = d
    K = 1. / (2 * N ** 0.5 * M ** 0.5)
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G


def gram_matrix_val(x, area, depth):
    F = x.reshape(area, depth)
    G = np.dot(F.T, F)
    return G


def style_layer_loss(a, x):
    _, h, w, d = a.shape
    M = h * w
    N = d
    A = gram_matrix_val(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def main(content, style):
    output = "{0}_{1}".format(style, content)
    c = os.path.join(CONTENT_DIR, content)
    s = os.path.join(STYLE_DIR, style)
    o = os.path.join(OUTOUT_DIR, output)

    print("Content: {}".format(c))
    content_img = load_image(c)
    print("Style: {}".format(s))
    style_img = load_image(s)
    init_img = noise_image(content_img)
    print("Output: {}".format(o))

    print("building vgg19...")
    net = build_vgg19(content_img)

    print("starting tf.session...")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("stylizing...")
    sess.run([net['input'].assign(content_img)])
    L_content = sum(map(lambda l: l[1] * content_layer_loss(sess.run(net[l[0]]), net[l[0]]), CONTENT_LAYERS))
    sess.run([net['input'].assign(style_img)])
    L_style = sum(map(lambda l: l[1] * style_layer_loss(sess.run(net[l[0]]), net[l[0]]), STYLE_LAYERS))
    L_total = L_content + STYLE_WEIGHT * L_style

    """minimize with lbfgs"""
    print("minimizing with lbfgs...")
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(L_total, method='L-BFGS-B',
                                                       options={'maxiter': ITERATION, 'disp': 1})
    sess.run(tf.global_variables_initializer())
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)
    result_img = sess.run(net['input'])

    """output"""
    print("outputing image...")
    save_image(o, result_img)

if __name__ == '__main__':
    mode = input("Input '1' to make just one image, '2' to make all images in the folder:")
    if mode == 1:
        print('*** make one! ***')
        c_image = input('Content image:')
        s_image = input('Style image:')
        main(c_image, s_image)
    else:
        print('*** make all! ***')
        content_list = list()
        style_list = list()
        for rt, dirs, files in os.walk(CONTENT_DIR):
            for f in files:
                content_list.append(f)
        for rt, dirs, files in os.walk(STYLE_DIR):
            for f in files:
                style_list.append(f)
        for c_image in content_list:
            for s_image in style_list:
                main(c_image, s_image)
