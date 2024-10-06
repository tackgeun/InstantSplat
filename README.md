
<h2 align="center"> <a href="https://arxiv.org/abs/2403.20309">InstantSplat: Sparse-view SfM-free <a href="https://arxiv.org/abs/2403.20309"> Gaussian Splatting in Seconds </a>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.20309) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kairunwen/InstantSplat) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://instantsplat.github.io/)[![X](https://img.shields.io/badge/-Twitter@Zhiwen%20Fan%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/WayneINR/status/1774625288434995219)  [![youtube](https://img.shields.io/badge/Demo_Video-E33122?logo=Youtube)](https://youtu.be/fxf_ypd7eD8) [![youtube](https://img.shields.io/badge/Tutorial_Video-E33122?logo=Youtube)](https://www.youtube.com/watch?v=JdfrG89iPOA&t=347s)
</h5>

<div align="center">
Modified implementation of InstantSplat for Supporting [MAST3r](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction).
</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Free-view Rendering](#free-view-rendering)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## TODO List
- [x] Support MAST3r for SfM
- [ ] Confidence-aware Point Cloud Downsampling

## Get Started

### Installation
1. Clone InstantSplat and download pre-trained model.
```bash
git clone --recursive https://github.com/NVlabs/InstantSplat.git
cd InstantSplat
git submodule update --init --recursive
```

2. Create the environment (or use pre-built docker), here we show an example using conda.
```bash
conda create -n instantsplat python=3.11 cmake=3.14.0
conda activate instantsplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install submodules/simple-knn
# modify the rasterizer
vim submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
'p_view.z <= 0.2f' -> 'p_view.z <= 0.001f' # line 154
pip install submodules/diff-gaussian-rasterization
```

3. Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd submodules/dust3r/croco/models/curope/
python setup.py build_ext --inplace
```

Alternative: use the pre-built docker image: pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
```
docker pull dockerzhiwen/instantsplat_public:2.0
```
if docker failed to produce reasonable results, try Installation step again within the docker.

### Usage
1. Data preparation
```
  <data_path>/<dataset-name>/<scene-name>/<#view_views>/images/
```

2. Command
```bash
  # InstantSplat train and output video (no GT reference, render by interpolation) using the following command.
  bash scripts/run_train_infer.sh

  # InstantSplat train and evaluate (with GT reference) using the following command.
  bash scripts/run_train_eval.sh
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [MASt3R](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{fan2024instantsplat,
        title={InstantSplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds},
        author={Zhiwen Fan and Wenyan Cong and Kairun Wen and Kevin Wang and Jian Zhang and Xinghao Ding and Danfei Xu and Boris Ivanovic and Marco Pavone and Georgios Pavlakos and Zhangyang Wang and Yue Wang},
        year={2024},
        eprint={2403.20309},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
      }
```
