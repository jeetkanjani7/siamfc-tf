
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.io
import sys
import os.path
import nbimporter


# In[2]:


sys.path.insert(0, '/home/jeetkanjani7/Tonbo/siamfc-tf/')


# In[3]:


from src.convolutional import set_convolutional
from src.crops import extract_crops_z, extract_crops_x, pad_frame, resize_images
from src.parse_arguments import parse_arguments


# In[4]:


pos_x_ph = tf.placeholder(tf.float64)
pos_y_ph = tf.placeholder(tf.float64)
z_sz_ph = tf.placeholder(tf.float64)

x_sz0_ph = tf.placeholder(tf.float64)
x_sz1_ph = tf.placeholder(tf.float64)
x_sz2_ph = tf.placeholder(tf.float64)


# In[5]:


_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True


# In[6]:


assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)


# In[7]:


def build_tracking_graph_nt(final_score_sz, design, env):
    filename = tf.placeholder(tf.string, [], name = 'filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file)

    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
    frame_sz = tf.shape(image)
    if design.pad_with_image_mean:
        avg_chan = tf.reduce_mean(image, axis=(0,1), name = 'avg_chan')
    else:
        avg_chan = None
    frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph,pos_y_ph, z_sz_ph, avg_chan)
    frame_padded_z = tf.cast(frame_padded_z, tf.float32)
    
    z_crops = extract_crops_z(image, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, design.exemplar_sz)
    frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan)
    frame_padded_x = tf.cast(frame_padded_x, tf.float32)
    
    x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph, 255)#design.search_sz)
    template_z, template_x, p_names_list, p_val_list = create_siamese(os.path.join(env.root_pretrained, design.net), x_crops, z_crops)
    template_z = tf.squeeze(template_z)
    template_z = tf.stack([template_z, template_z, template_z])
    
    scores = match_templates(template_z, template_x, p_names_list, p_val_list)
    score_up = tf.image.resize_images(scores, (final_score_sz, final_score_sz), method=tf.image.ResizeMethod.BICUBIC, align_corners= True)
    return filename, image, template_z, score_up


# In[8]:


def create_siamese(net_path, net_x, net_z):
    print("hi")
    params_names_list, params_values_list = import_from_matconvnet(net_path)
    print params_names_list
    for i in xrange(_num_layers):
        print '> Layer '+str(i+1)
        # conv
        conv_W_name = _find_params('conv'+str(i+1)+'f', params_names_list)[0]
        conv_b_name = _find_params('conv'+str(i+1)+'b', params_names_list)[0]
        print '\t\tCONV: setting '+conv_W_name+' '+conv_b_name
        print '\t\tCONV: stride '+str(_conv_stride[i])+', filter-group '+str(_filtergroup_yn[i])
        conv_W = params_values_list[params_names_list.index(conv_W_name)]
        conv_b = params_values_list[params_names_list.index(conv_b_name)]
        # batchnorm
        if _bnorm_yn[i]:
            bn_beta_name = _find_params('bn'+str(i+1)+'b', params_names_list)[0]
            bn_gamma_name = _find_params('bn'+str(i+1)+'m', params_names_list)[0]
            bn_moments_name = _find_params('bn'+str(i+1)+'x', params_names_list)[0]
            print '\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name
            bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
            bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
            bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
            bn_moving_mean = bn_moments[:,0]
            bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
        
        # set up conv "block" with bnorm and activation 
        net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i],                             bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,                             filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i],                             scope='conv'+str(i+1), reuse=False)
        
        # notice reuse=True for Siamese parameters sharing
        net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i],                             bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,                             filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i],                             scope='conv'+str(i+1), reuse=True)    
        
        # add max pool if required
        if _pool_stride[i]>0:
            print '\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i])
            net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
            net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))



    return net_z, net_x, params_names_list, params_values_list


# In[9]:


def import_from_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_do_mat = mat.get('net')
    
    params = net_do_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list =  [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_value_list = [params_values[p] for p in range(params_values.size)]
    return params_names_list, params_value_list


# In[10]:


def match_templates(net_z, net_x, params_names_list, params_values_list):
    # finalize network
    # z, x are [B, H, W, C]
    net_z = tf.transpose(net_z, perm=[1,2,0,3])
    net_x = tf.transpose(net_x, perm=[1,2,0,3])
    # z, x are [H, W, B, C]
    Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
    Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
    # assert B==Bx, ('Z and X should have same Batch size')
    # assert C==Cx, ('Z and X should have same Channels number')
    net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
    net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
    # final is [1, Hf, Wf, BC]
    net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)
    # final is [B, Hf, Wf, C]
    net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
    # final is [B, Hf, Wf, 1]
    if _bnorm_adjust:
        bn_beta = params_values_list[params_names_list.index('fin_adjust_bnb')]
        bn_gamma = params_values_list[params_names_list.index('fin_adjust_bnm')]
        bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
        bn_moving_mean = bn_moments[:,0]
        bn_moving_variance = bn_moments[:,1]**2
        net_final = tf.layers.batch_normalization(net_final, beta_initializer=tf.constant_initializer(bn_beta),
                                                gamma_initializer=tf.constant_initializer(bn_gamma),
                                                moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
                                                moving_variance_initializer=tf.constant_initializer(bn_moving_variance),
                                                training=False, trainable=False)

    return net_final


# In[11]:


def _find_params(x, params):
    matching = [s for s in params if x in s]
    assert len(matching)==1, ('Ambiguous param name found')    
    return matching


