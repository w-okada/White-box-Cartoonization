'''
Source code for CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
by Xinrui Wang and Jinze yu
'''

import sys
import os
print(sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), 'selective_search'))
print(sys.path)

import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
import numpy as np
import argparse
import network 
import loss

from tqdm import tqdm

import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
# tf.get_logger().setLevel(logging.ERROR)

from guided_filter import guided_filter

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_image",   default = 'dataset/photo_movie_ss_r',      type = str)    
    parser.add_argument("--cartoon_image", default = 'dataset/cartoon_movie_ss_r',    type = str)        
    parser.add_argument("--pre_save_dir",  default = 'pretrain/saved_models',         type = str)        
    parser.add_argument("--vgg_19",        default = './vgg19_no_fc.npy',             type = str)        
    parser.add_argument("--patch_size",    default = 256,                             type = int)
    parser.add_argument("--batch_size",    default = 16,                              type = int)     
    parser.add_argument("--total_iter",    default = 100000,                          type = int)
    parser.add_argument("--adv_train_lr",  default = 2e-4,                            type = float)
    parser.add_argument("--gpu_fraction",  default = 0.5,                             type = float)
    parser.add_argument("--save_dir",      default = 'train_cartoon',                 type = str)
    parser.add_argument("--use_enhance",   default = False)

    args = parser.parse_args()
    
    return args



def train(args):
    

    input_photo = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_superpixel = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_cartoon = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    
    output = network.unet_generator(input_photo)
    output = guided_filter(input_photo, output, r=1)

    
    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)
    
    d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake, 
                                             scale=1, patch=True, name='disc_gray')
    d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake, 
                                             scale=1, patch=True, name='disc_blur')


    vgg_model = loss.Vgg19(args.vgg_19)
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(output)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]
    
    photo_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_photo, vgg_output))/(h*w*c)
    superpixel_loss = tf.reduce_mean(tf.losses.absolute_difference\
                                     (vgg_superpixel, vgg_output))/(h*w*c)
    recon_loss = photo_loss + superpixel_loss
    tv_loss = loss.total_variation_loss(output)
    
    g_loss_total = 1e4*tv_loss + 1e-1*g_loss_blur + g_loss_gray + 2e2*recon_loss
    d_loss_total = d_loss_blur + d_loss_gray

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name] 
    
    
    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('photo_loss', photo_loss)
    tf.summary.scalar('superpixel_loss', superpixel_loss)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('d_loss_gray', d_loss_gray)
    tf.summary.scalar('g_loss_gray', g_loss_gray)
    tf.summary.scalar('d_loss_blur', d_loss_blur)
    tf.summary.scalar('g_loss_blur', g_loss_blur)
    tf.summary.scalar('d_loss_total', d_loss_total)
    tf.summary.scalar('g_loss_total', g_loss_total)
      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        g_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(g_loss_total, var_list=gene_vars)
        
        d_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(d_loss_total, var_list=disc_vars)
           
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    train_writer = tf.summary.FileWriter(args.save_dir+'/train_log')
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)

    out_path = os.path.join(args.save_dir, 'saved_models')
    # print("out:"+out_path)
    # saver.restore(sess, tf.train.latest_checkpoint(out_path))
    with tf.device('/device:GPU:0'):

        sess.run(tf.global_variables_initializer())
        print(args.pre_save_dir)
        saver.restore(sess, tf.train.latest_checkpoint(args.pre_save_dir))

        photo_dir = args.photo_image
        photo_list = utils.load_image_list(photo_dir)

        cartoon_dir = args.cartoon_image
        cartoon_list = utils.load_image_list(cartoon_dir)

        for total_iter in tqdm(range(args.total_iter)):
            photo_batch = utils.next_batch(photo_list, args.batch_size)
            cartoon_batch = utils.next_batch(cartoon_list, args.batch_size)
        
            inter_out = sess.run(output, feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: photo_batch,
                                                    input_cartoon: cartoon_batch})

            '''
            adaptive coloring has to be applied with the clip_by_value 
            in the last layer of generator network, which is not very stable.
            to stabiliy reproduce our results, please use power=1.0
            and comment the clip_by_value function in the network.py first
            If this works, then try to use adaptive color with clip_by_value.
            '''
            if args.use_enhance:
                superpixel_batch = utils.selective_adacolor(inter_out, power=1.2)
            else:
                superpixel_batch = utils.simple_superpixel(inter_out, seg_num=200)
                
            _, g_loss, r_loss = sess.run([g_optim, g_loss_total, recon_loss],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: superpixel_batch,
                                                    input_cartoon: cartoon_batch})

            _, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: superpixel_batch,
                                                    input_cartoon: cartoon_batch})


            train_writer.add_summary(train_info, total_iter)
            
            if np.mod(total_iter+1, 100) == 0:

                print('Iter: {}, d_loss: {}, g_loss: {}, recon_loss: {}'.\
                        format(total_iter, d_loss, g_loss, r_loss))
                if np.mod(total_iter+1, 2000 ) == 0:
                    saver.save(sess, out_path+"/model", 
                               write_meta_graph=False, global_step=total_iter)
                     
                    photo = utils.next_batch(photo_list, args.batch_size)
                    cartoon = utils.next_batch(cartoon_list, args.batch_size)

                    result = sess.run(output, feed_dict={input_photo: photo, 
                                                            input_superpixel: photo,
                                                            input_cartoon: cartoon})


                    utils.write_batch_image(result, os.path.join(args.save_dir, 'images'), 
                                            str(total_iter)+'_result.jpg', 4)
                    utils.write_batch_image(photo, os.path.join(args.save_dir, 'images'), 
                                            str(total_iter)+'_photo.jpg', 4)

            
if __name__ == '__main__':
    
    args = arg_parser()
    train(args)  
   