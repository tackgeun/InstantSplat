import os
import shutil
import torch
import numpy as np
import argparse
import time

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

def get_intrinsics(scene): 
    K = torch.zeros((scene.n_imgs, 3, 3)) 
    focals = scene.get_focals().view(scene.n_imgs, -1) 
    K[:, 0, 0] = focals[:, 0] 
    K[:, 1, 1] = focals[:, -1] 
    K[:, :2, 2] = scene.get_principal_points() 
    K[:, 2, 2] = 1 
    return K

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--model_path", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.07)
    parser.add_argument("--niter", type=int, default=500)
    parser.add_argument("--focal_avg", action="store_true")

    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--img_base_path", type=str, default="/home/workspace/datasets/ins/tantsplat/Tanks/Barn/24_views")

    parser.add_argument("--scene_graph", type=str, default='swin-3')

    return parser

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    n_views = args.n_views
    img_base_path = args.img_base_path
    scene_graph = args.scene_graph
    img_folder_path = os.path.join(img_base_path, "images")
    os.makedirs(img_folder_path, exist_ok=True)
    
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    ##########################################################################################################################################################################################

    train_img_list = sorted(os.listdir(img_folder_path))
    assert len(train_img_list)==n_views, f"Number of images ({len(train_img_list)}) in the folder ({img_folder_path}) is not equal to {n_views}"
    images, ori_size = load_images(img_folder_path, size=512)
    print("ori_size", ori_size)

    start_time = time.time()
    ##########################################################################################################################################################################################
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    output_colmap_path=img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    scene = sparse_global_alignment(train_img_list, pairs, os.path.join('output', 'cache'),
                                    model, lr1=lr, niter1=niter, lr2=0, niter2=None, device=device,
                                    opt_depth=False, shared_intrinsics=True,
                                    matching_conf_thr=1.0)
        
    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d, depthmaps, confidence_masks = scene.get_dense_pts3d()
    img_size = confidence_masks[0].size()
    pts3d = to_numpy([_pts.cpu().view(img_size[0], img_size[1], -1) for _pts in pts3d])
    confidence_masks = to_numpy([_conf.cpu() >= 1.0 for _conf in confidence_masks])
    intrinsics = to_numpy(get_intrinsics(scene))
    ##########################################################################################################################################################################################
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

    # save colmap format
    print('save colmap cameras and images')
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", np.array(focals.cpu()))

    print('concat pts3d')
    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    print('concat images')
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    print('store ply')

    # avoid process kill due to memory size
    import gc
    del pts3d
    del imgs
    del confidence_masks
    del depthmaps
    del pts_4_3dgs_all
    gc.collect()
    
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
