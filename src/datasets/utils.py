import cv2
import os
import torch
import numpy as np
import open3d as o3d

INTRINSIC = {
    'master': np.asarray([
        [969.48345947265625,    0,                  1024.9678955078125],
        [0,                     968.99578857421875, 781.4013671875],
        [0,                     0,                  1]]),
    'sub': np.asarray([
        [972.07073974609375,    0,                  1021.4869384765625  ],
        [0,                     971.651123046875,   780.25439453125     ],
        [0,                     0,                  1                   ]
    ])
}


class SequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, 
                 resource=['radar','master_image','master_image','sub_image','sub_depth'], **kwargs) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        # load transformation matrix
        with open(os.path.join(seq_path, 'calib.txt')) as f:
            self.calib = eval(f.readline())

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'radar'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx: int):
        result = {}
        if 'radar' in self.resource:
            result['radar'] = np.load(os.path.join(
                self.seq_path, 'radar', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'master_image' in self.resource:
            result['master_image'] = cv2.imread(os.path.join(
                self.seq_path, 'image', 'master', 'frame_{}.png'.format(idx+self.skip_head)))
        if 'sub_image' in self.resource:
            result['sub_image'] = cv2.imread(os.path.join(
                self.seq_path, 'image', 'sub', 'frame_{}.png'.format(idx+self.skip_head)))
        if 'master_depth' in self.resource:
            result['master_depth'] = np.load(os.path.join(
                self.seq_path, 'depth_pcl', 'master', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'sub_depth' in self.resource:
            result['sub_depth'] = np.load(os.path.join(
                self.seq_path, 'depth_pcl', 'sub', 'frame_{}.npy'.format(idx+self.skip_head)))
        result['mesh'] = np.load(os.path.join(
            self.seq_path, 'mesh', 'frame_{}.npz'.format(idx+self.skip_head)))
            
        return result

def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def project_pcl(pcl, trans_mat=None, cam='master', image_size=np.array([1536,2048])):
    if trans_mat is not None:
        pcl = (pcl - trans_mat['t']) @ trans_mat['R']
    pcl_2d = ((pcl/pcl[:,2:3]) @ INTRINSIC[cam].T)[:,:2]
    pcl_2d = np.floor(pcl_2d).astype(int)
    pcl_2d[:, [0, 1]] = pcl_2d[:, [1, 0]]
    # filter out the points exceeding the image size
    pcl_2d = np.where(pcl_2d<image_size-1, pcl_2d, image_size-1)
    pcl_2d = np.where(pcl_2d>0, pcl_2d, 0)
    
    return pcl_2d


def filter_pcl(bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """
    upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
    lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
    lower_bound[2] += offset

    mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
        target_pcl[:, 0] <= upper_bound[0])
    mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
        target_pcl[:, 1] <= upper_bound[1])
    mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
        target_pcl[:, 2] <= upper_bound[2])
    index = mask_x & mask_y & mask_z
    return target_pcl[index]


def pad_pcl(pcl, num_points, return_choices=False):
    if pcl.shape[0] > num_points:
        r = np.random.choice(pcl.shape[0], size=num_points, replace=False)
    else:
        repeat, residue = num_points // pcl.shape[0], num_points % pcl.shape[0]
        r = np.random.choice(pcl.shape[0], size=residue, replace=False)
        r = np.concatenate([np.arange(pcl.shape[0]) for _ in range(repeat)] + [r], axis=0)
    if return_choices:
        return pcl[r, :], r
    return pcl[r, :]


def crop_image(joints:np.ndarray, image:np.ndarray, trans_mat:dict=None, visual:bool=False, margin:float=0.2, square:bool=False, cam='master'):
    """
    Crop the person area of image
    """
    # transform the joints to camera coordinate
    if trans_mat is not None:
        joints = (joints - trans_mat['t']) @ trans_mat['R']
    joint_max = joints.max(axis=0) + margin
    joint_min = joints.min(axis=0) - margin
    # get 3d bounding box from joints
    box_3d = np.array([
        [joint_min[0], joint_min[1], joint_min[2]],
        [joint_min[0], joint_min[1], joint_max[2]],
        [joint_min[0], joint_max[1], joint_max[2]],
        [joint_min[0], joint_max[1], joint_min[2]],
        [joint_max[0], joint_max[1], joint_max[2]],
        [joint_max[0], joint_min[1], joint_max[2]],
        [joint_max[0], joint_max[1], joint_min[2]],
        [joint_max[0], joint_min[1], joint_min[2]],
    ])
    # project 3d bounding box to 2d image plane
    box_2d = project_pcl(box_3d, cam=cam)
    box_min = box_2d.min(0)
    box_max = box_2d.max(0)
    if square:
        size = box_max - box_min
        diff = abs(size[0] - size[1])//2
        if size[0] > size[1]:
            box_max[1] += diff
            box_min[1] -= diff
        elif size[0] < size[1]:
            box_max[0] += diff
            box_min[0] -= diff
    box_max = np.where(box_max<image.shape[:2], box_max, image.shape[:2])
    box_min = np.where(box_min>0, box_min, 0)
    # crop image
    crop_img = image[box_min[0]:box_max[0], box_min[1]:box_max[1]]

    if visual:
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.rectangle(image, box_min[::-1], box_max[::-1], (0, 0, 255), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    return crop_img

def trans_mat_2_tensor(trans_mat):
    trans_mat_array = np.hstack((trans_mat['R'], np.array([trans_mat['t']]).T))
    trans_mat_array = np.vstack((trans_mat_array, [0,0,0,1]))
    return trans_mat_array

def trans_mat_2_dict(trans_mat):
    trans_mat = copy2cpu(trans_mat)
    trans_mat_dict = {
        'R': trans_mat[:3, :3],
        't': trans_mat[:3, 3]
    }
    return trans_mat_dict

def get_rgb_value(pcl, image, visual=False, ret_image=False):
    pcl_2d = project_pcl(pcl)
    pcl_color = image[pcl_2d[:, 0], pcl_2d[:, 1]]
    pcl_with_feature = np.hstack((pcl, pcl_color/255))

    if visual:
        image[pcl_2d[:, 0], pcl_2d[:, 1]] = [0, 255, 0]
        cv2.namedWindow('img', 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    return pcl_with_feature

def convert_square_image(image):
    """
     convert to square with slice
    """
    img_h, img_w, img_c = image.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = int(abs(img_w - img_h)/2)
        image = image.transpose((1, 0, 2)).copy() if img_w < img_h else image
        background = np.full((long_side, long_side, img_c), 255, np.uint8)
        background[loc: loc + short_side] = image[...]
        image = background.transpose((1, 0, 2)).copy() if img_w < img_h else background
    return image

def rodrigues_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(torch.zeros(
        [total_size], device="cuda"))  # for broadcasting
    Ks_1 = torch.stack([zero, -u[:, 2],  u[:, 1]], axis=1)  # row 1
    Ks_2 = torch.stack([u[:, 2],  zero, -u[:, 0]], axis=1)  # row 2
    Ks_3 = torch.stack([-u[:, 1],  u[:, 0],  zero], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    identity_mat = torch.autograd.Variable(
        torch.eye(3, device="cuda").repeat(total_size, 1, 1))
    Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
        (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:, None, None], identity_mat, Rs)

    return R.reshape(batch_size, -1)

def gen_random_indices(max_random_num, min_random_num=0, random_ratio=1., random_num=True, size=0):
    # generate x% of max_random_num random indices
    pb = np.random.rand() if random_num else 1.
    num_indices = np.floor(pb*random_ratio*(max_random_num-min_random_num)) if not size else size
    indices = np.random.choice(np.arange(min_random_num, max_random_num), replace=False, size=int(num_indices))
    return np.sort(indices)

def mosh_pose_transform(trans, root_orient, root_joint, trans_mat):
    mosh_offset = root_joint - trans
    new_trans = trans_mat['R'] @ (trans + mosh_offset) + trans_mat['t'] - mosh_offset
    orient_mat = trans_mat['R'] @ cv2.Rodrigues(root_orient)[0]
    new_orient = cv2.Rodrigues(orient_mat)[0]
    return new_trans, new_orient