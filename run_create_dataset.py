from multiprocessing import Pool
import time
import os
import cv2
import numpy as np
import zipfile

import torch
from src.datasets.utils import HuMManSequenceLoader, copy2cpu, crop_image, depth_2_pcl, filter_pcl, project_pcl
from src.modeling._smpl import SMPL

def zip_files(source_path: str, target_path: str):
    zip = zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(source_path):
        fpath = path.replace(source_path, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

def make_dirs(save_path: str):
    if not os.path.exists(os.path.join(save_path, 'crop_image', 'kinect_000')):
        os.makedirs(os.path.join(save_path, 'crop_image', 'kinect_000'))
    if not os.path.exists(os.path.join(save_path, 'crop_image', 'kinect_001')):
        os.makedirs(os.path.join(save_path, 'crop_image', 'kinect_001'))
    # if not os.path.exists(os.path.join(save_path, 'depth', 'kinect_000')):
    #     os.makedirs(os.path.join(save_path, 'depth', 'kinect_000'))
    # if not os.path.exists(os.path.join(save_path, 'depth', 'kinect_001')):
    #     os.makedirs(os.path.join(save_path, 'depth', 'kinect_001'))
    os.system('rm -r {}'.format(os.path.join(save_path, 'depth_pcl')))
    if not os.path.exists(os.path.join(save_path, 'depth_pcl', 'kinect_000')):
        os.makedirs(os.path.join(save_path, 'depth_pcl', 'kinect_000'))
    if not os.path.exists(os.path.join(save_path, 'depth_pcl', 'kinect_001')):
        os.makedirs(os.path.join(save_path, 'depth_pcl', 'kinect_001'))
    # if not os.path.exists(os.path.join(save_path, 'mesh')):
    #     os.makedirs(os.path.join(save_path, 'mesh'))
    # if not os.path.exists(os.path.join(save_path, 'bounding_box', 'kinect_000')):
    #     os.makedirs(os.path.join(save_path, 'bounding_box', 'kinect_000'))
    # if not os.path.exists(os.path.join(save_path, 'bounding_box', 'kinect_001')):
    #     os.makedirs(os.path.join(save_path, 'bounding_box', 'kinect_001'))

def save_data(seq_path):
    print(seq_path, 'start', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    seq_loader = HuMManSequenceLoader(seq_path, resource=['master_image','sub_image','master_depth','sub_depth'])
    smpl_model = SMPL()

    for idx in range(len(seq_loader)):
        # print(idx)
        # if idx < 1788:
        #     continue
        frame = seq_loader[idx]
        smpl_para = dict(frame['mesh'])
        pose = np.concatenate((smpl_para['transl'], smpl_para['global_orient'], smpl_para['body_pose']))
        betas = smpl_para['betas']
        mesh = smpl_model(torch.from_numpy(pose[None]).float(), torch.from_numpy(betas[None]).float())
        joints = copy2cpu(mesh['joints'][0])
        trans_mat = seq_loader.calib
        master_img = frame['master_image'] / [255, 255, 255]
        # crop_img = crop_image(joints, master_img, trans_mat=trans_mat['kinect_color_000'], margin=0.1, 
        #                         square=True, intrinsic=trans_mat['kinect_color_000']['K'], return_box=False)
        mas_depth_img = frame['master_depth']
        mas_depth_pcl = depth_2_pcl(mas_depth_img, trans_mat['kinect_depth_000']['K'])
        mas_depth_pcl = mas_depth_pcl @ trans_mat['kinect_depth_000']['R'].T + trans_mat['kinect_depth_000']['t']
        
        mas_crop_pcl = filter_pcl(joints, mas_depth_pcl, bound=0.1)
        
        pcl_2d = project_pcl(mas_crop_pcl, trans_mat=trans_mat['kinect_color_000'], 
                             intrinsic=trans_mat['kinect_color_000']['K'], image_size=np.array([1080, 1920]))
        master_pcl = np.concatenate((mas_crop_pcl, master_img[pcl_2d[:,0], pcl_2d[:,1]]), 1)
        # joints_2d = project_pcl(joints, trans_mat=trans_mat['kinect_color_000'], intrinsic=trans_mat['kinect_color_000']['K'])
        # for i in range(-5, 5):
        #     for j in range(-5, 5):
        #         master_img[pcl_2d[:,0]+i, pcl_2d[:,1]+j] = [0, 255, 0]
        # cv2.namedWindow('image0', 0)
        # cv2.resizeWindow("image0", 960, 480)
        # cv2.imshow('image0', master_img)
        # cv2.waitKey(0)

        # master_bbox = np.array([mas_box_min, mas_box_max])
        # master_pcl = frame['master_pcl']
        # master_color = master_img.reshape(len(master_pcl), 3) / [255, 255, 255]
        # master_pcl = filter_pcl(mesh_joint, np.hstack((master_pcl, master_color)), 0.2, 0.21)
        # master_img = cv2.resize(crop_img, (224, 224))
        
        sub_img = frame['sub_image'] / [255, 255, 255]
        # crop_img = crop_image(joints, sub2_img, trans_mat=trans_mat['kinect_color_001'], margin=0.1, 
        #                         square=True, intrinsic=trans_mat['kinect_color_001']['K'], return_box=False)
        
        sub_depth_img = frame['sub_depth']
        sub_depth_pcl = depth_2_pcl(sub_depth_img, trans_mat['kinect_depth_001']['K'])
        sub_depth_pcl = sub_depth_pcl @ trans_mat['kinect_depth_001']['R'].T + trans_mat['kinect_depth_001']['t']
        
        sub_crop_pcl = filter_pcl(joints, sub_depth_pcl, bound=0.1)
        
        pcl_2d = project_pcl(sub_crop_pcl, trans_mat=trans_mat['kinect_color_001'], 
                             intrinsic=trans_mat['kinect_color_001']['K'], image_size=np.array([1080, 1920]))
        sub_pcl = np.concatenate((sub_crop_pcl, sub_img[pcl_2d[:,0], pcl_2d[:,1]]), 1)
        
        # joints_2d = project_pcl(sub_crop_pcl, trans_mat=trans_mat['kinect_color_001'], intrinsic=trans_mat['kinect_color_001']['K'])
        # for i in range(-5, 5):
        #     for j in range(-5, 5):
        #              sub_img[pcl_2d[:,0]+i, pcl_2d[:,1]+j] = [0, 255, 0]
        # cv2.namedWindow('image1', 0)
        # cv2.resizeWindow('image1', 960, 480)
        # cv2.imshow('image1', sub_img)
        # cv2.waitKey(0)

        # sub2_bbox = np.array([sub_box_min, sub_box_max])
        # sub2_pcl = frame['sub2_pcl']
        # sub2_color = sub2_img.reshape(len(sub2_pcl), 3) / [255, 255, 255]
        # sub2_pcl = filter_pcl(mesh_joint, np.hstack((sub2_pcl, sub2_color)), 0.2, 0.21)
        # sub2_img = cv2.resize(crop_img, (224, 224))
        # sub2_depth_img = frame['sub2_depth']
        
        # o3d_plot([o3d_pcl(sub2_pcl[:,:3]-mesh_joint[0], color=[0,0,1]), o3d_pcl(mesh_joint-mesh_joint[0], color=[0,1,0])])
        
        # np.save(os.path.join(save_path, 'radar', 'frame_{}'.format(idx)), arbe_data)
        # cv2.imwrite(os.path.join(seq_path, 'crop_image', 'kinect_000', '%06d.png' % (idx*6)), master_img)
        # np.save(os.path.join(save_path, 'bounding_box', 'master', 'frame_{}'.format(idx)), master_bbox)
        np.save(os.path.join(seq_path, 'depth_pcl', 'kinect_000', '%06d.npy' % (idx*6)), master_pcl)
        # cv2.imwrite(os.path.join(save_path, 'depth', 'master', 'frame_{}.png'.format(idx)), master_depth_img)
        # cv2.imwrite(os.path.join(seq_path, 'crop_image', 'kinect_001', '%06d.png' % (idx*6)), sub2_img)
        # np.save(os.path.join(save_path, 'bounding_box', 'sub', 'frame_{}'.format(idx)), sub2_bbox)
        np.save(os.path.join(seq_path, 'depth_pcl', 'kinect_001', '%06d.npy' % (idx*6)), sub_pcl)
        # cv2.imwrite(os.path.join(save_path, 'depth', 'sub', 'frame_{}.png'.format(idx)), sub2_depth_img)
        # np.savez(os.path.join(save_path, 'mesh', 'frame_{}'.format(idx)), **mesh)
        
    print(seq_path, 'end', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

def main():
    train_paths = []
    test_paths = []
    train_path = '/media/nesc525/perple2/HuMMan/train'
    test_path = '/media/nesc525/perple2/HuMMan/test'

    train_paths += [os.path.join(train_path, p) for p in os.listdir(train_path)]
    test_paths += [os.path.join(test_path, p) for p in os.listdir(test_path)]

    pool = Pool(40)
    
    # for i, seq_path in enumerate(train_paths):
    #     if seq_path != '/media/nesc525/perple2/HuMMan/train/p100062_a000224':
    #         continue
    #     make_dirs(seq_path)
    #     # pool.apply_async(save_data, (seq_path,))
    #     save_data(seq_path)

    
    # for i, seq_path in enumerate(test_paths):
    # #     # if i == 1:
    # #     #     continue
    #     make_dirs(seq_path)
    #     pool.apply_async(save_data, (seq_path,))
    #     # save_data(seq_path)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main()