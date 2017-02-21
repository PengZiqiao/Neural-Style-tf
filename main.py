import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
from argparse import ArgumentParser

CONTENT_DIR = './images'
STYLE_DIR = './styles'
OUTPUT_DIR = './results'
OUTPUT_IMG = 'results.jpg'

layer_list = (
    [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)],
    [('conv1_1', 0.5), ('conv2_1', 1.), ('conv3_1', 1.5), ('conv4_1', 3.), ('conv5_1', 4.)],
    [('conv1_1', 1.), ('conv2_1', 1.5), ('conv3_1', 2.), ('conv4_1', 2.5), ('conv5_1', 3.)]
)
STYLE_LAYERS = layer_list[0]
CONTENT_LAYERS = [('conv4_2', 1.)]
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def build_parser():
    """通过命令行设置参数
    """
    desc = '---- 基于Tensorflow的风格迁移实例 ----'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--content', type=str,
                        help='内容图像文件名， e.g. Tuebingen.jpg')
    parser.add_argument('--style', type=str,
                        help='风格图像文件名， e.g. StarryNight.jpg')
    parser.add_argument('--weight', type=float, default=500,
                        help='风格权重， e.g. 500', )
    parser.add_argument('--noise', type=float, default=0.6,
                        help='初始图像的随机噪声比例， e.g. 0.6')
    parser.add_argument('--iter', type=int, default=250,
                        help='训练迭代次数， e.g. 800')
    args = parser.parse_args()
    return args


class NeuralStyle:
    """风格迁移类
    """

    def __init__(self, content, style, output="result.jpg", style_weight=1 / 8e-4, noise_ratio=0.7, iterations=250):
        self.content = content  # 内容图像文件名
        self.style = style  # 风格图像文件名
        self.output = output  # 输出图像文件名
        self.style_weight = style_weight  # 风格权重
        self.noise_ratio = noise_ratio  # 初始图像的随机噪声比例
        self.iterations = iterations  # 训练迭代次数
        pass

    def get_shape(self, path):
        """确定图像尺寸
        以content的图像尺寸为标准
        """
        img = scipy.misc.imread(path)
        height, width, channel = img.shape
        return height, width, channel

    def read_image(self, path, shape):
        """读入图像
        path: 文件地址
        shape: 图像尺寸
        """
        h, w, d = shape
        img = scipy.misc.imread(path)
        # shape (h, w, d) to (1, h, w, d)
        img = img[np.newaxis, :h, :w, :d]
        # Input to the VGG model expects the mean to be subtracted.
        image = img - MEAN_VALUES
        return image

    def save_image(self, path, image):
        """保存图像
        path: 存放地址
        img: 保存的图片
        """
        # Output should add back the mean.
        img = image + MEAN_VALUES
        # shape (1, h, w, d) to (h, w, d)
        img = img[0]
        img = np.clip(img, 0, 255).astype('uint8')
        scipy.misc.imsave(path, img)

    def noise_image(self, img, shape, noise):
        """生成初始图像
        img: 用于和nosie混合的图
        shape: 图像尺寸
        noise: 随机噪声的比重
        """
        h, w, d = shape
        init = np.random.uniform(-20, 20, (1, h, w, d)).astype('float32')
        init = noise * init + (1. - noise) * img
        return init

    def build_vgg19(self, shape):
        """构建神经网络"""

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

        vgg_rawnet = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')  # 请先将vgg19放置在项目根目录
        vgg_layers = vgg_rawnet['layers'][0]

        h, w, d = shape
        net = dict()
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

    def loss(self, sess, net, content_img, style_img):
        """计算loss"""

        def sum_content_losses():
            def content_layer_loss(p, x):
                # _, h, w, d = p.shape
                # M = h * w
                # N = d
                # K = 1. / (2 * N ** 0.5 * M ** 0.5)
                K = 0.5
                loss = K * tf.reduce_sum(tf.pow((x - p), 2))
                return loss

            sess.run(net['input'].assign(content_img))
            content_loss = 0.
            for layer, weight in CONTENT_LAYERS:
                p = sess.run(net[layer])
                x = net[layer]
                p = tf.convert_to_tensor(p)
                content_loss += content_layer_loss(p, x) * weight
            return content_loss

        def sum_style_losses():
            def style_layer_loss(a, x):
                _, h, w, d = a.shape
                M = h * w
                N = d
                A = gram_matrix(a, M, N)
                G = gram_matrix(x, M, N)
                K = 1. / (4 * N ** 2 * M ** 2)
                loss = K * tf.reduce_sum(tf.pow((G - A), 2))
                return loss

            def gram_matrix(x, area, depth):
                F = tf.reshape(x, (area, depth))
                G = tf.matmul(tf.transpose(F), F)
                return G

            sess.run(net['input'].assign(style_img))
            style_loss = 0.
            for layer, weight in STYLE_LAYERS:
                a = sess.run(net[layer])
                x = net[layer]
                a = tf.convert_to_tensor(a)
                style_loss += style_layer_loss(a, x) * weight
            return style_loss

        L_content = sum_content_losses()
        L_style = sum_style_losses()
        L_total = L_content + self.style_weight * L_style
        return L_total

    def draw(self):
        """主程序
        工作流程：
        读入图片 >>> 构建神经网络 >>> 计算loss >>> 迭代训练 >>> 保存结果
        """
        # 将文件路径补充完整
        content_path = os.path.join(CONTENT_DIR, self.content)
        style_path = os.path.join(STYLE_DIR, self.style)
        output_path = os.path.join(OUTPUT_DIR, self.output)
        # 确定图像尺寸
        shape = self.get_shape(content_path)
        # 读入图片
        print(">>>正在读取文件...")
        print(">>>content: {}, style: {}".format(self.content, self.style))
        content_img = self.read_image(content_path, shape)
        style_img = self.read_image(style_path, shape)
        init_img = self.noise_image(content_img, shape, self.noise_ratio)
        # 构建神经网络
        print(">>>正在构建神经网络...")
        net = self.build_vgg19(shape)  # TODO
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 计算loss
        print(">>>正在计算loss值")
        L_total = self.loss(sess, net, content_img, style_img)
        # minimize with lbfgs
        print(">>>开始使用 L-BFGS-B 方法进行迭代({}次)...".format(self.iterations))
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(L_total, method='L-BFGS-B',
                                                           options={'maxiter': self.iterations, 'disp': 1})
        sess.run(tf.global_variables_initializer())
        sess.run(net['input'].assign(init_img))
        optimizer.minimize(sess)
        result_img = sess.run(net['input'])
        # 保存图片
        print(">>>迭代完成，正在保存图片，请不要关闭程序...")
        self.save_image(output_path, result_img)


if __name__ == '__main__':
    option = build_parser()
    artist = NeuralStyle(option.content, option.style)
    artist.draw()
