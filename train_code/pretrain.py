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
from tqdm import tqdm

import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
# tf.get_logger().setLevel(logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_image",  default = 'dataset/photo_movie_ss_r', type = str)    
    parser.add_argument("--patch_size",   default = 256,                        type = int)
    parser.add_argument("--batch_size",   default = 16,                         type = int)     
    parser.add_argument("--total_iter",   default = 50000,                      type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4,                       type = float)
    parser.add_argument("--gpu_fraction", default = 0.5,                        type = float)
    parser.add_argument("--save_dir",     default = 'pretrain')

    args = parser.parse_args()
    
    return args

def train(args):
    input_photo = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    
    output = network.unet_generator(input_photo)
    
    recon_loss = tf.reduce_mean(tf.losses.absolute_difference(input_photo, output))

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(recon_loss, var_list=gene_vars)
        
        
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)

    os.makedirs(args.save_dir, exist_ok=True)
    #with tf.device('/device:GPU:0'):
    with tf.device('device:XLA_GPU:0'):
        sess.run(tf.global_variables_initializer())
        photo_dir = args.photo_image
        photo_list = utils.load_image_list(photo_dir)

        for total_iter in tqdm(range(args.total_iter)):
            photo_batch = utils.next_batch(photo_list, args.batch_size)
                
            _, r_loss = sess.run([optim, recon_loss], feed_dict={input_photo: photo_batch})

            if np.mod(total_iter+1, 50) == 0:

                print('pretrain, iter: {}, recon_loss: {}'.format(total_iter, r_loss))
                if np.mod(total_iter+1, 500 ) == 0:
                    saver.save(sess, os.path.join(args.save_dir, 'saved_models/model'), 
                               write_meta_graph=False, global_step=total_iter)
                     
                    photo = utils.next_batch(photo_list, args.batch_size)
                    result = sess.run(output, feed_dict={input_photo: photo})
                   
                    utils.write_batch_image(result, os.path.join(args.save_dir,'/images'), 
                                            str(total_iter)+'_result.jpg', 4)
                    utils.write_batch_image(photo, os.path.join(args.save_dir,'/images'),
                                            str(total_iter)+'_photo.jpg', 4)
                    

 
            
if __name__ == '__main__':
    
    args = arg_parser()
    train(args)  
   