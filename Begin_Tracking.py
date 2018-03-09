
# coding: utf-8

# In[1]:


import sys 
import os

from scipy.misc import imsave
import numpy as np
from PIL import Image
import scipy as sc
import nbimporter
import tensorflow as tf
from IPython.display import HTML 
import imp
import time
import imageio
import matplotlib.pyplot as plt
import tensorboard


# In[2]:


sys.path.insert(0, '/home/jeetkanjani7/Tonbo/siamfc-tf/')


# In[3]:


from src.tracker import tracker
#import src.siamese as siam
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.visualization import show_frame, show_crops, show_scores



# In[4]:


sys.path.insert(1 ,'/home/jeetkanjani7/Tonbo/siamfc-tf/Notebook/')


# In[5]:



import Siamese_nt as siam_nt


# In[6]:


def init_video():
    video_folder = "/home/jeetkanjani7/Tonbo/siamfc-tf/data/dataset/tc_Ball_ce2/"
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [ (video_folder) + s for s in frame_name_list] 
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asanyarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]
        
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter = ',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal'
    return gt, frame_name_list, frame_sz, n_frames 


# In[7]:


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    
    if xA < xB and yA < yB:
        interarea = (xB - xA) * (yB - yA)
        
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        iou = interarea/(boxAArea + boxBArea - interarea)
    else:
        iou = 0
    assert iou >= 0
    assert iou<=1.01
    
    return iou


# In[8]:


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


# In[9]:


def compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l,4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)
    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    precision_auc = np.trapz(precisions_ths)    
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


# In[10]:


def init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames


# In[11]:



hp, evaluation, run, env, design = parse_arguments()
final_score_sz = hp.response_up * (design.score_sz -1) + 1
filename, image, templates_z, scores = siam_nt.build_tracking_graph_nt(final_score_sz, design, env)


# In[12]:


gt, frame_name_list, _ , _ = init_video(env, evaluation, evaluation.video)
pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])


# In[13]:


num_frames = np.size(frame_name_list)
# stores tracker's output for evaluation
bboxes = np.zeros((num_frames,4))

scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
# cosine window to penalize large displacements    
hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
penalty = np.transpose(hann_1d) * hann_1d
penalty = penalty / np.sum(penalty)

context = design.context*(target_w+target_h)
z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

# thresholds to saturate patches shrinking/growing
min_z = hp.scale_min * z_sz
max_z = hp.scale_max * z_sz
min_x = hp.scale_min * x_sz
max_x = hp.scale_max * x_sz

    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

run_opts = {}


# In[16]:


def update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


# In[ ]:



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
      
    # save first frame position (from ground-truth)
    bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h                

    image_, templates_z_ = sess.run([image, templates_z], feed_dict={siam_nt.pos_x_ph: pos_x,                                                                     siam_nt.pos_y_ph: pos_y,                                                                     siam_nt.z_sz_ph: z_sz,                                                                     filename: frame_name_list[0]})
                    
    new_templates_z_ = templates_z_

    t_start = time.time()

    # Get an image from the queue
    for i in range(1, num_frames):        
        scaled_exemplar = z_sz * scale_factors
        scaled_search_area = x_sz * scale_factors
        scaled_target_w = target_w * scale_factors
        scaled_target_h = target_h * scale_factors
        image_, scores_ = sess.run(
            [image, scores],
            feed_dict={
                siam_nt.pos_x_ph: pos_x,
                siam_nt.pos_y_ph: pos_y,
                siam_nt.x_sz0_ph: scaled_search_area[0],
                siam_nt.x_sz1_ph: scaled_search_area[1],
                siam_nt.x_sz2_ph: scaled_search_area[2],                    templates_z: np.squeeze(templates_z_),
                    filename: frame_name_list[i],
                }, **run_opts)
        scores_ = np.squeeze(scores_)
        # penalize change of scale
        scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
        scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
        # find scale with highest peak (after penalty)
        new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
        # update scaled sizes
        x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]        
        target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
        target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
        # select response with new_scale_id
        score_ = scores_[new_scale_id,:,:]
        score_ = score_ - np.min(score_)
        score_ = score_/np.sum(score_)
        # apply displacement penalty
        score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
        imageio.imwrite("./scores/"+str(i)+".png", score_)
        pos_x, pos_y = update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
        # update the target representation with a rolling average
        if hp.z_lr>0:
            new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                siam_nt.pos_x_ph: pos_x,
                                                                siam_nt.pos_y_ph: pos_y,
                                                                siam_nt.z_sz_ph: z_sz,
                                                                image: image_
                                                                })

            templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
            
        # update template patch size
        z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]
            
        if run.visualization:
            #show_frame(image_, bboxes[i,:], 1)  
            print("")

    t_elapsed = time.time() - t_start
    speed = num_frames/t_elapsed

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads) 

    # from tensorflow.python.client import timeline
    # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    # trace_file = open('timeline-search.ctf.json', 'w')
    # trace_file.write(trace.generate_chrome_trace_format())

plt.close('all')


# In[15]:


writer = tf.summary.FileWriter("/graph/1")


# In[ ]:


writer.add_graph(sess.graph)

