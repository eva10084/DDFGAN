from datetime import datetime
import json
import numpy as np
import os
import random

#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf

import data_loader
import losses
import model
import cv2

import csv
from stats_func import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''source_train_pth = './datactmr/datalist/training_B.txt'
target_train_pth = './datactmr/datalist/training_A_'
source_val_pth = './datactmr/datalist/validation_B.txt'
target_val_pth = './datactmr/datalist/validation_A_'  
'''

source_train_pth = 'road/source_train.txt'
target_train_pth = 'road/target_train_'
source_val_pth = 'road/source_val.txt'
target_val_pth = 'road/target_val_'


evaluation_interval = 10
save_interval = 300
num_cls = 4
keep_rate_value=0.75
is_training_value=True

BATCH_SIZE = 5

class PCH:
    """The PCH module."""

    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step,
                 checkpoint_dir, do_flipping, skip, kfold):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._source_train_pth = source_train_pth
        self._target_train_pth = target_train_pth
        self._source_val_pth = source_val_pth
        self._target_val_pth = target_val_pth
        self._num_cls = num_cls
        tf.compat.v1.disable_eager_execution()
        self.keep_rate = tf.compat.v1.placeholder(tf.float32, shape=())
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self._pool_size = pool_size
        self._size_before_crop = 512
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, str(kfold))
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip
        self._kfold = kfold

        self.fake_images_A = np.zeros(
            (self._pool_size, BATCH_SIZE, model.IMG_HEIGHT, model.IMG_WIDTH, 1)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, BATCH_SIZE, model.IMG_HEIGHT, model.IMG_WIDTH, 1)
        )


    def model_setup(self):
        # 搭建网络

        # 图像
        # 在 TensorFlow 中创建一个占位符（placeholder）节点
        # 占位符是一个可以在 TensorFlow 计算图中使用的空节点，它没有实际的值，但是可以在运行时被传入具体的数据来填充。
        self.input_a = tf.compat.v1.placeholder(
            tf.float32, [   #  指定了张量中元素的数据类型为 32 位浮点型
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                3           # 指定了张量的形状，其中第一维的大小为 None，表示该维度大小可以是任意非负整数。
            ], name="input_A")     # 该张量的名称为 "input_A"
        # 表示该张量包含的是 model.IMG_WIDTH x model.IMG_HEIGHT 的彩色图像，其中 3 表示三个颜色通道（红、绿、蓝）
        self.input_b = tf.compat.v1.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                3
            ], name="input_B")
        self.fake_pool_A = tf.compat.v1.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_A")
        self.fake_pool_B = tf.compat.v1.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_B")
        # 标注，512，512，4
        self.gt_a = tf.compat.v1.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                4
            ], name="gt_A")
        self.gt_b = tf.compat.v1.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                4
            ], name="gt_B")

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        # 它返回全局步数张量，该张量是一个标量张量，用于跟踪已经执行的训练步数。
        # 通常在优化器中与全局步数张量一起使用，以调整学习率或执行其他操作。

        self.num_fake_inputs = 0

        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name="lr")
        # 占位符节点通常用于接受一个实数，表示模型的学习率。该节点的值在模型训练期间会被不断地更新，以实现不同训练阶段的学习率调整。
        self.learning_rate_seg = tf.compat.v1.placeholder(tf.float32, shape=[], name="lr_seg")
        # 分割器学习率

        # 创建一个TensorFlow的标量汇总(summary)来记录当前GAN学习率的值，名称为“lr_gan”，值为self.learning_rate。
        # 此汇总(summary)用于在TensorBoard中可视化和跟踪学习率的变化。
        self.lr_gan_summ = tf.summary.scalar("lr_gan", self.learning_rate)
        self.lr_seg_summ = tf.summary.scalar("lr_seg", self.learning_rate_seg)

        # 上述定义输入图，以及一些初始化格式的操作，将placeholder组成input
        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
            'gt_a': self.gt_a,    # 源域标注，无目标域标注
        }

        # 调用model文件中的函数，其后得到一系列结果
        outputs = model.get_outputs(
            inputs, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.pred_mask_b = outputs['pred_mask_b']
        self.pred_mask_fake_b = outputs['pred_mask_fake_b']
        self.pre_mask_real_a = outputs['pre_mask_real_a']
        self.prob_fea_fake_b_is_real = outputs['prob_fea_fake_b_is_real']
        self.prob_fea_b_is_real = outputs['prob_fea_b_is_real']

        self.prob_real_a_aux = outputs['prob_real_a_aux']
        self.prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
        self.prob_fake_pool_a_aux_is_real = outputs['prob_fake_pool_a_aux_is_real']
        self.prob_cycle_a_is_real = outputs['prob_cycle_a_is_real']
        self.prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']

        self.fea_A_separate_B = outputs['fea_A_separate_B']
        self.fea_B_separate_A = outputs['fea_B_separate_A']
        self.fea_FA_separate_B = outputs['fea_FA_separate_B']
        self.fea_FB_separate_A = outputs['fea_FB_separate_A']

#######################
        # 这段代码实现了生成器的预测结果的评估。
        # 首先，使用 pixel_wise_softmax_2 函数将生成器预测结果进行 softmax 处理，
        # 并使用 tf.argmax 函数提取最大值索引作为紧凑的预测标签。
        # 接着，使用 tf.argmax 函数提取 ground truth 标签的最大值索引。
        # 最后，使用 tf.confusion_matrix 函数计算混淆矩阵，以评估模型的预测性能。
        self.predicter_fake_b = pixel_wise_softmax_2(self.pred_mask_fake_b)
        self.compact_pred_fake_b = tf.argmax(self.predicter_fake_b, 3)
        self.compact_y_fake_b = tf.argmax(self.gt_a, 3)
        self.confusion_matrix_fake_b = tf.confusion_matrix(tf.reshape(self.compact_y_fake_b, [-1]),
                                                           tf.reshape(self.compact_pred_fake_b, [-1]),
                                                           num_classes=self._num_cls)

        self.predicter_b = pixel_wise_softmax_2(self.pred_mask_b)
        self.compact_pred_b = tf.argmax(self.predicter_b, 3)
        self.compact_y_b = tf.argmax(self.gt_b, 3)
        self.confusion_matrix_b = tf.confusion_matrix(tf.reshape(self.compact_y_b, [-1]),
                                                      tf.reshape(self.compact_pred_b, [-1]), num_classes=self._num_cls)

        self.dice_b_arr =  dice_eval(self.compact_pred_b, self.gt_b,self._num_cls)
        self.dice_b_mean = tf.reduce_mean(self.dice_b_arr)


    def compute_losses(self):
        # 损失函数计算

        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=tf.expand_dims(self.input_a[:,:,:,1], axis=3), generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=tf.expand_dims(self.input_b[:,:,:,1], axis=3), generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)
        lsgan_loss_f = losses.lsgan_loss_generator(self.prob_fea_b_is_real)
        lsgan_loss_a_aux = losses.lsgan_loss_generator(self.prob_fake_a_aux_is_real)

        losssea=0.01*losses.l1_loss(self.fea_A_separate_B)
        lossseb = 0.01*losses.l1_loss(self.fea_B_separate_A)
        lossseaf=0.01*losses.l1_loss(self.fea_FA_separate_B)
        losssebf = 0.01*losses.l1_loss(self.fea_FB_separate_A)
        dif_loss=losssea+lossseb+lossseaf+losssebf


        ce_loss_b, dice_loss_b = losses.task_loss(self.pred_mask_fake_b, self.gt_a)
        ce_loss_a, dice_loss_a = losses.task_loss(self.pre_mask_real_a, self.gt_a)

        l2_loss_b = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/s_B/' in v.name or '/e_B/' in v.name])


        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        self.loss_f_weight = tf.placeholder(tf.float32, shape=[], name="loss_f_weight")
        self.loss_f_weight_summ = tf.summary.scalar("loss_f_weight", self.loss_f_weight)
        #seg_loss_B = ce_loss_b + dice_loss_b + ce_loss_a + dice_loss_a + l2_loss_b + 0.1*g_loss_B + self.loss_f_weight*lsgan_loss_f + 0.1*lsgan_loss_a_aux
        seg_loss_B = ce_loss_b + dice_loss_b  + l2_loss_b + 0.1*g_loss_B + self.loss_f_weight*lsgan_loss_f + 0.1*lsgan_loss_a_aux
        seg_loss_A = ce_loss_a + dice_loss_a  +  l2_loss_b ++ 0.1*g_loss_A


        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_A_aux = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_cycle_a_aux_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_aux_is_real,
        )
        d_loss_A = d_loss_A + d_loss_A_aux
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )
        d_loss_F = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_fea_fake_b_is_real,
            prob_fake_is_real=self.prob_fea_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        optimizer_seg = tf.train.AdamOptimizer(self.learning_rate_seg)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if '/d_A/' in var.name]
        d_B_vars = [var for var in self.model_vars if '/d_B/' in var.name]
        e_c_vars = [var for var in self.model_vars if '/e_c/' in var.name]
        e_cs_vars = [var for var in self.model_vars if '/e_cs/' in var.name]
        e_ct_vars = [var for var in self.model_vars if '/e_ct/' in var.name]
        de_B_vars = [var for var in self.model_vars if '/de_B/' in var.name]
        de_A_vars = [var for var in self.model_vars if '/de_A/' in var.name]
        de_c_vars = [var for var in self.model_vars if '/de_c/' in var.name]
        s_B_vars = [var for var in self.model_vars if '/s_B/' in var.name]
        d_F_vars = [var for var in self.model_vars if '/d_F/' in var.name]
        e_dB_vars = [var for var in self.model_vars if '/e_dB/' in var.name]
        e_dA_vars = [var for var in self.model_vars if '/e_dA/' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.dif_trainer = optimizer.minimize(dif_loss, var_list=e_dB_vars + e_dA_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=de_A_vars+de_c_vars+e_c_vars+e_cs_vars+e_dB_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=de_B_vars+de_c_vars+e_c_vars+e_ct_vars+e_dA_vars)
        self.s_B_trainer = optimizer_seg.minimize(seg_loss_B, var_list=e_c_vars+e_ct_vars+s_B_vars)
        self.s_A_trainer = optimizer_seg.minimize(seg_loss_A, var_list=e_c_vars + e_cs_vars + s_B_vars)
        self.d_F_trainer = optimizer.minimize(d_loss_F, var_list=d_F_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
        self.dif_loss_summ = tf.summary.scalar("dif_loss", dif_loss)
        self.ce_B_loss_summ = tf.summary.scalar("ce_B_loss", ce_loss_b)
        self.dice_B_loss_summ = tf.summary.scalar("dice_B_loss", dice_loss_b)
        self.l2_B_loss_summ = tf.summary.scalar("l2_B_loss", l2_loss_b)
        self.s_B_loss_summ = tf.summary.scalar("s_B_loss", seg_loss_B)
        self.s_A_loss_summ = tf.summary.scalar("s_A_loss", seg_loss_A)
        self.s_B_loss_merge_summ = tf.summary.merge([self.ce_B_loss_summ, self.dice_B_loss_summ, self.l2_B_loss_summ, self.s_B_loss_summ,self.s_A_loss_summ])
        self.d_F_loss_summ = tf.summary.scalar("d_F_loss", d_loss_F)

    def fake_image_pool(self, num_fakes, fake, fake_pool):

        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        # 模型训练

        # 从数据集文件夹加载数据集
#        TTrain = self._target_train_pth + str(self._kfold) +'.txt'     # 训练源域图片名称
#        TVal = self._target_val_pth + str(self._kfold) +'.txt'        # 训练时目标域图片名称
        TTrain = self._source_train_pth
        TVal = self._source_val_pth
        # input的四元组：源域和目标域的图像（512，512，3）3通道彩色，gt_i, gt_j  源域和目标域的标注（512，512，4）4个one-hot编码。
        self.inputs = data_loader.load_data(self._source_train_pth, TTrain, True)   # 调用data_load文件（源域、目标域，均为txt）
        self.inputs_val = data_loader.load_data(self._source_val_pth, TVal, True)   # 验证集文件（源域，目标域）（均为txt）
        # 都是二维图片了，进行了切分


        # 搭建网络
        self.model_setup()

        # 损失函数计算
        self.compute_losses()

        # 初始化全局变量
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)

        with open(self._source_train_pth, 'r') as fp:   # 打开文件夹
            rows_s = fp.readlines()
        with open(TTrain, 'r') as fp:
            rows_t = fp.readlines()

        with open(TVal, 'r') as fp:
            rows_t_val = fp.readlines()

        max_images = max(len(rows_s), len(rows_t))

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)


            # 从上一个模型的checkpoint加载模型
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir,sess.graph)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # 训练循环
            curr_lr_seg = 0.001
            cnt = -1
            val_dice=0

            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch ", epoch)

                curr_lr = self._base_lr

                if epoch < 5:
                    loss_f_weight_value = 0.0
                elif epoch < 7:
                    loss_f_weight_value = 0.1 * (epoch - 4.0) / (7.0 - 4.0)
                else:
                    loss_f_weight_value = 0.1


                if epoch > 0 and epoch%2==0:
                    curr_lr_seg = np.multiply(curr_lr_seg, 0.9)

                max_inter = np.uint16(np.floor(max_images/BATCH_SIZE))

                for i in range(0, max_inter):
                    cnt += 1

                    print("Processing batch {}/{}".format(i, max_inter))
					

                    images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                    inputs = {
                        'images_i': images_i,
                        'images_j': images_j,
                        'gts_i': gts_i,
                        'gts_j': gts_j,
                    }
                    images_i_val, images_j_val, gts_i_val, gts_j_val = sess.run(self.inputs_val)
                    inputs_val = {
                        'images_i_val': images_i_val,
                        'images_j_val': images_j_val,
                        'gts_i_val': gts_i_val,
                        'gts_j_val': gts_j_val,
                    }

                    # 优化 G_A network   生成器
                    _, fake_B_temp, summary_str = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.gt_a:
                                inputs['gts_i'],
                            self.learning_rate: curr_lr,
                            self.keep_rate:keep_rate_value,
                            self.is_training:is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # 优化 D_B network  判别器
                    _, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the S_B network  分割器
                    _, summary_str = sess.run(
                        [self.s_B_trainer, self.s_B_loss_merge_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.gt_a:
                                inputs['gts_i'],
                            self.learning_rate_seg: curr_lr_seg,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }

                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the S_A network
                    _, summary_str = sess.run(
                        [self.s_A_trainer, self.s_B_loss_merge_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.gt_a:
                                inputs['gts_i'],
                            self.learning_rate_seg: curr_lr_seg,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }

                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.gt_a: inputs['gts_i'],
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the D_F network
                    _, summary_str = sess.run(
                        [self.d_F_trainer, self.d_F_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the Dif network
                    _, summary_str = sess.run(
                        [self.dif_trainer, self.dif_loss_summ],
                        feed_dict={
                            self.input_a:
                                    inputs['images_i'],
                            self.input_b:
                                    inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                     )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    summary_str_gan, summary_str_seg, summary_str_lossf = sess.run([self.lr_gan_summ, self.lr_seg_summ, self.loss_f_weight_summ],
                             feed_dict={
                                 self.learning_rate: curr_lr,
                                 self.learning_rate_seg: curr_lr_seg,
                                 self.loss_f_weight: loss_f_weight_value,
                             })

                    writer.add_summary(summary_str_gan, epoch * max_inter + i)
                    writer.add_summary(summary_str_seg, epoch * max_inter + i)
                    writer.add_summary(summary_str_lossf, epoch * max_inter + i)
                    writer.flush()
                    self.num_fake_inputs += 1

                # save the best val model
                print("Processing val {}".format(i))
                max_inter_val = np.uint16(np.floor(len(rows_t_val) / BATCH_SIZE))
                val_dice_list = []
                for m in range(0, max_inter_val):
                    np_b_mean = sess.run([self.dice_b_mean], feed_dict = {self.input_b: inputs_val['images_j_val'],self.gt_b:inputs_val['gts_j_val'], self.is_training: False, self.keep_rate: 1.0})
                    val_dice_list.append(np_b_mean)
                val_dice_mean=np.mean(val_dice_list)
                print(len(val_dice_list), val_dice_mean)
                if(val_dice_mean > val_dice):
                    val_dice = val_dice_mean
                    saver.save(sess, os.path.join(self._output_dir, "pch"), global_step=cnt)

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)

def main(log_dir, config_filename, checkpoint_dir, skip, kfold):

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    # 一个用于清空当前默认计算图的工具函数。
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()

    # 定义超参数
    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = False  # 是否需要恢复之前的训练进度
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002  # 开始的学习率
    max_step = int(config['max_step']) if 'max_step' in config else 100   # 最大步
    do_flipping = bool(config['do_flipping'])   # 图片是否翻转

    # 设置pch的init
    pch_model = PCH(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step,
                              checkpoint_dir, do_flipping, skip, kfold)

    pch_model.train()


if __name__ == '__main__':
    #i for K_fold
   # for i in range(0, 2):
    main(log_dir='output', config_filename='configs/exp_01.json', checkpoint_dir='', skip=True, kfold=1)
