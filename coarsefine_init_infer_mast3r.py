import os
import shutil
import torch
import numpy as np
import argparse
import time
import math 
import random

import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r", 'dust3r')))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dust3r.inference import inference
#from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images

def bchw_order(t):
    t = t.unsqueeze(0)
    t = t.permute(0, 3, 1, 2)
    return t

def numpy_order(t):
    t = t.squeeze(0)
    t = t.permute(1,2,0)
    return to_numpy(t)

def get_intrinsics(scene): 
    K = torch.zeros((scene.n_imgs, 3, 3)) 
    focals = scene.get_focals().view(scene.n_imgs, -1) 
    K[:, 0, 0] = focals[:, 0] 
    K[:, 1, 1] = focals[:, -1] 
    K[:, :2, 2] = scene.get_principal_points() 
    K[:, 2, 2] = 1 
    return K


def compute_scene_graph(process_name, img_list, img_folder_path, args, lr, niter, lr_reproj, niter_reproj, scene_graph, init={}):
    #import pdb; pdb.set_trace()
    device = args.device
    n_views = args.n_views
    batch_size = args.batch_size
    len_img = len(img_list)

    start_time = time.time()
    images, ori_size = load_images([os.path.join(img_folder_path, img_name) for img_name in img_list], size=512)
    print("ori_size", ori_size)
    
    randint = random.randint(0, 10000000)
    cache_path = os.path.join('output', f'cache{randint}')
    while os.path.exists(cache_path):
        randint = random.randint(0, 10000000)
        cache_path = os.path.join('output', f'cache{randint}')
    
    os.mkdir(cache_path)

    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    #import pdb; pdb.set_trace()
    scene = sparse_global_alignment(img_list, pairs, cache_path,
                                    model, lr1=lr, niter1=niter, lr2=lr_reproj, niter2=niter_reproj, device=device,
                                    opt_depth=False, shared_intrinsics=True,
                                    matching_conf_thr=1.0, init=init)

    imgs = to_numpy(scene.imgs)
    focals = to_numpy(scene.get_focals())
    poses = to_numpy(scene.get_im_poses())
    quats = to_numpy(scene.get_quats())
    trans = to_numpy(scene.get_trans())
    log_sizes = to_numpy(scene.get_log_sizes())
    pts3d, depthmaps, confidence_masks = scene.get_dense_pts3d()
    img_size = confidence_masks[0].size()
    #pts3d = to_numpy([_pts.cpu().view(img_size[0], img_size[1], -1) for _pts in pts3d])
    #confidence_masks = to_numpy([_conf.cpu() >= 1.0 for _conf in confidence_masks])
    pts3d = [_pts.cpu().view(img_size[0], img_size[1], -1) for _pts in pts3d]
    confidence_masks = [_conf.cpu() >= 1.0 for _conf in confidence_masks]

    intrinsics = to_numpy(get_intrinsics(scene))
    end_time = time.time()
    print(process_name + f"Time taken for {len_img} frames: {end_time-start_time} seconds")

    shutil.rmtree(cache_path)

    return imgs, focals, poses, quats, trans, log_sizes, pts3d, depthmaps, confidence_masks, intrinsics, ori_size

import torch

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Args:
        R: Rotation matrix (3x3 torch tensor)
    Returns:
        quat: Quaternion (torch tensor of shape [4])
    """
    # Compute trace
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2  # S=4*qx
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2  # S=4*qy
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # S=4*qz
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    quat = torch.tensor([qw, qx, qy, qz])
    return quat

def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion to a rotation matrix.
    Args:
        quat: Quaternion (torch tensor of shape [4])
    Returns:
        R: Rotation matrix (3x3 torch tensor)
    """
    qw, qx, qy, qz = quat
    R = torch.zeros(3, 3)
    R[0, 0] = 1 - 2*(qy**2 + qz**2)
    R[0, 1] = 2*(qx*qy - qz*qw)
    R[0, 2] = 2*(qx*qz + qy*qw)
    R[1, 0] = 2*(qx*qy + qz*qw)
    R[1, 1] = 1 - 2*(qx**2 + qz**2)
    R[1, 2] = 2*(qy*qz - qx*qw)
    R[2, 0] = 2*(qx*qz - qy*qw)
    R[2, 1] = 2*(qy*qz + qx*qw)
    R[2, 2] = 1 - 2*(qx**2 + qy**2)
    return R

def slerp_quaternion(quat1, quat2, t):
    """
    Perform spherical linear interpolation (slerp) between two quaternions.
    Args:
        quat1: First quaternion (torch tensor of shape [4])
        quat2: Second quaternion (torch tensor of shape [4])
        t: Interpolation factor (float)
    Returns:
        quat_interp: Interpolated quaternion (torch tensor of shape [4])
    """
    # Normalize quaternions
    quat1 = quat1 / quat1.norm()
    quat2 = quat2 / quat2.norm()
    
    # Compute the cosine of the angle between the two vectors.
    cos_half_theta = torch.dot(quat1, quat2)
    
    # If cos_half_theta < 0, the interpolation will take the long way around the sphere.
    # To fix this, one quaternion must be negated.
    if cos_half_theta < 0.0:
        quat2 = -quat2
        cos_half_theta = -cos_half_theta

    # If the quaternions are close, use linear interpolation
    if cos_half_theta > 0.9995:
        quat_interp = quat1 * (1.0 - t) + quat2 * t
        quat_interp = quat_interp / quat_interp.norm()
        return quat_interp

    # Compute the angle between the quaternions
    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    # If sin_half_theta is close to zero, use linear interpolation
    if torch.abs(sin_half_theta) < 0.001:
        quat_interp = quat1 * 0.5 + quat2 * 0.5
        quat_interp = quat_interp / quat_interp.norm()
        return quat_interp

    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    quat_interp = quat1 * ratio_a + quat2 * ratio_b
    quat_interp = quat_interp / quat_interp.norm()
    return quat_interp

def interpolate_c2w(c2w1, c2w2, t):
    """
    Interpolate between two camera-to-world matrices (c2w1 and c2w2) using linear interpolation.
    Args:
        c2w1: First camera-to-world matrix (4x4 torch tensor).
        c2w2: Second camera-to-world matrix (4x4 torch tensor).
        t: Interpolation factor, between 0 and 1 (float).
    Returns:
        Interpolated camera-to-world matrix (4x4 torch tensor).
    """
    # Extract rotation matrices
    R1 = c2w1[:3, :3]
    R2 = c2w2[:3, :3]

    # Convert rotation matrices to quaternions
    quat1 = rotation_matrix_to_quaternion(R1)
    quat2 = rotation_matrix_to_quaternion(R2)

    # Slerp between the two quaternions
    quat_interp = slerp_quaternion(quat1, quat2, t)

    # Convert quaternion back to rotation matrix
    R_interp = quaternion_to_rotation_matrix(quat_interp)

    # Linearly interpolate translation components
    translation_interp = (1 - t) * c2w1[:3, 3] + t * c2w2[:3, 3]

    # Assemble the new c2w matrix
    c2w_interp = torch.eye(4)
    c2w_interp[:3, :3] = R_interp
    c2w_interp[:3, 3] = translation_interp

    return c2w_interp

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--model_path", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.07)
    parser.add_argument("--niter", type=int, default=500)
    parser.add_argument("--lr_reproj", type=float, default=0.014)
    parser.add_argument("--niter_reproj", type=int, default=200)

    parser.add_argument("--focal_avg", action="store_true")

    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--img_base_path", type=str, default="/home/workspace/datasets/ins/tantsplat/Tanks/Barn/24_views")

    parser.add_argument("--scene_graph", type=str, default='swin-3')
    parser.add_argument("--scene_graph_local", type=str, default='swin-3')

    parser.add_argument("--subsample_frame", type=int, default=1)
    parser.add_argument("--subsample_res", type=float, default=1.0)
    parser.add_argument("--skip_frame", type=int, default=30)

    return parser


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    model_path = args.model_path    
    schedule = args.schedule
    img_base_path = args.img_base_path
    device = args.device
    n_views = args.n_views    

    ##########################################################################################################################################################################################        
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

    img_folder_path = os.path.join(img_base_path, "images")
    os.makedirs(img_folder_path, exist_ok=True)

    train_img_list = sorted(os.listdir(img_folder_path))
    assert len(train_img_list)==n_views, f"Number of images ({len(train_img_list)}) in the folder ({img_folder_path}) is not equal to {n_views}"

    output_colmap_path=img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    ##########################################################################################################################################################################################        
    # COMPUTE GLOBAL STRUCTURE
    ##########################################################################################################################################################################################        
    K = args.skip_frame
    global_img_list = [value for index, value in enumerate(train_img_list) if (index+1) % K == 1]

    imgs, focals, poses, quats, trans, log_sizes, pts3d, depthmaps, confidence_masks, intrinsics, ori_size = compute_scene_graph(
        '[global graph] ',
        global_img_list,
        img_folder_path,
        args,
        args.lr,
        args.niter,
        args.lr_reproj,
        args.niter_reproj,
        args.scene_graph
    )

    if args.subsample_frame > 1:
        ssf = args.subsample_frame
        print(f'subsampling frames every {ssf} frame.')
        ssf = args.subsample_frame 
        pts3d = [value for index, value in enumerate(pts3d) if (index+1) % ssf == 1]
        confidence_masks = [value for index, value in enumerate(confidence_masks) if (index+1) % ssf == 1]
        imgs = [value for index, value in enumerate(imgs) if (index+1) % ssf == 1]
    elif args.subsample_res > 1.0:
        scale_f = 1.0/args.subsample_res
        pts3d = [numpy_order(F.interpolate(bchw_order(value), scale_factor=scale_f, mode='bilinear')) for value in pts3d]
        imgs = [numpy_order(F.interpolate(bchw_order(torch.Tensor(value)), scale_factor=scale_f, mode='bilinear')) for value in imgs]   
        confidence_masks = [to_numpy(F.interpolate(value.float().unsqueeze(0).unsqueeze(0), scale_factor=scale_f).squeeze(0).squeeze(0).bool()) for value in confidence_masks]

    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)

    print('concat pts3d')
    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    print('concat images')
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)

    # avoid process kill due to memory size
    import gc
    del pts3d
    del imgs
    del confidence_masks
    #del depthmaps
    gc.collect()
    
    print('store ply')    
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)

    ##########################################################################################################################################################################################        
    # COMPUTE LOCAL STRUCTURE
    ##########################################################################################################################################################################################        
    for i in range(0, int(math.ceil(len(train_img_list)/K))):

        s_quats = torch.Tensor(quats[i]).to(device=args.device)
        s_trans = torch.Tensor(trans[i]).to(device=args.device)
        s_sizes = torch.Tensor(log_sizes[i]).to(device=args.device)
        s_intrin = torch.Tensor(intrinsics[i]).to(device=args.device)
        s_pose = torch.Tensor(poses[i])

        if (i+1)*K + 1 <= len(train_img_list):
            last_idx = (i+1)*K + 1
            e_quats = torch.Tensor(quats[i+1]).to(device=args.device)
            e_trans = torch.Tensor(trans[i+1]).to(device=args.device)
            e_sizes = torch.Tensor(log_sizes[i+1]).to(device=args.device)
            e_intrin = torch.Tensor(intrinsics[i+1]).to(device=args.device)
            e_pose = torch.Tensor(poses[i+1])
        else:
            last_idx =  len(train_img_list)
            e_quats = torch.Tensor(quats[i]).to(device=args.device)
            e_trans = torch.Tensor(trans[i]).to(device=args.device)
            e_sizes = torch.Tensor(log_sizes[i]).to(device=args.device)
            e_intrin = torch.Tensor(intrinsics[i]).to(device=args.device)
            e_pose = torch.Tensor(poses[i])

        local_img_list = train_img_list[i*K:last_idx]

        local_intrinsics = []
        local_poses = []
        local_focals = []

        init_dict = dict()
        for k, _img in enumerate(local_img_list):
            interp_intrin = s_intrin * (K-k)/K + e_intrin * k/(K)
            interp_quats = s_quats * (K-k)/K + e_quats * k/(K)
            interp_trans = s_trans * (K-k)/K + e_trans * k/(K)
            interp_sizes = s_sizes * (K-k)/K + e_sizes * k/(K)
            interp_pose = interpolate_c2w(s_pose, e_pose, k/K)
            init_dict[_img] = {'intrinsics': interp_intrin, 'quat': interp_quats, 'tran': interp_trans, 'log_size': interp_sizes}

            local_intrinsics.append(to_numpy(interp_intrin))
            local_poses.append(to_numpy(interp_pose))
            local_focals.append(focals[0])
        
        #     if k == 0 or (k == len(local_img_list)-1 and last_idx == (i+1)*K + 1):
        #         init_dict[_img]['fixed'] = True # insert anchor
        #         if k == 0:
        #             init_dict[_img]['depthmap'] = depthmaps[i].to(device=args.device)
        #         elif (k == len(local_img_list)-1 and last_idx == (i+1)*K + 1):
        #             init_dict[_img]['depthmap'] = depthmaps[i+1].to(device=args.device)

        # local_imgs, local_focals, local_poses, local_quats, local_trans, local_sizes, \
        #     local_pts3d, local_depthmaps, local_confidence_masks, local_intrinsics, ori_size \
        #         = compute_scene_graph(
        #             f'[local graph {i}] ',
        #             local_img_list,
        #             img_folder_path,
        #             args,
        #             args.lr,
        #             args.niter,
        #             0.0, #args.lr_reproj, # avoid focal update
        #             0, #args.niter_reproj,
        #             args.scene_graph_local,
        #             init=init_dict
        #)

        if i == 0:
            # [i*K, ... , (i+1)*K, (i+1)*K + 1]
            all_intrinsics = local_intrinsics
            all_poses = local_poses
            all_focals = local_focals
        else:
            # [i*K, ... , (i+1)*K] [(i+1)*K + 1]
            cur_len = len(all_intrinsics)-1
            all_intrinsics = np.concatenate((all_intrinsics[:cur_len], local_intrinsics), axis=0)
            all_poses = np.concatenate((all_poses[:cur_len], local_poses), axis=0)
            all_focals = np.concatenate((all_focals[:cur_len], local_focals), axis=0)

    ##########################################################################################################################################################################################        
    # SAVE ALL INTRINSIC AND EXTRINSIC CAMERA PARAMETERS
    ##########################################################################################################################################################################################        

    np.save(output_colmap_path + "/focal.npy", all_focals)

    # save camera and image data with colmap format
    print('save colmap cameras and images')
    save_colmap_cameras(ori_size, all_intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(all_poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)
