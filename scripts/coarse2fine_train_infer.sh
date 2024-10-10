#! /bin/bash
DATA_ROOT_DIR="./data"

#GPU_ID=1

#DATASETS=(
#    kist	
#    #sora
#    )

#SCENES=(
#    #entrance
#    entrance2l8
#    #entrance2l8p288
#    #santorini
#    #L8
#    )

#N_VIEWS=(
#    750
#    #430
#    )

GPU_ID=$1
DATASETS=$2
SCENES=$3
N_VIEWS=$4
skip=$5
graph=$6
graphlocal=$7
subsample=$8
subsample_res=$9


# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=40000
pose_lr=1x
#subsample=5
#skip=30
#graph=swin-3
#graphlocal=swin-5
#subsample_res=2

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            # SOURCE_PATH must be Absolute path
            SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views
            MODEL_PATH="./output/infer/${DATASET}/${SCENE}/${N_VIEW}_views_Scene${graph}_${graphlocal}_Skip${skip}_Sub${subsample}-Res${subsample_res}_${gs_train_iter}Iter_${pose_lr}PoseLR/"
	    echo $MODEL_PATH
            # # ----- (1) MAST3r_coarse_geometric_initialization -----
            CMD_D1="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./coarsefine_init_infer_mast3r.py \
            --img_base_path ${SOURCE_PATH} \
            --n_views ${N_VIEW}  \
            --focal_avg \
	    --scene_graph ${graph}
	    --scene_graph_local ${graphlocal}
	    --subsample_frame ${subsample}
	    --subsample_res ${subsample_res}
	    --skip_frame ${skip}
            "

            # # ----- (2) Train: jointly optimize pose -----
	    
            CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_joint.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --optim_pose \
            "

            # ----- (3) Render interpolated pose & output video -----
            CMD_RI="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./render_by_interp.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --eval \
            --get_video \
            "


            echo "========= ${SCENE}: MAST3r_coarse_geometric_initialization ========="
            eval $CMD_D1
            echo "========= ${SCENE}: Train: jointly optimize pose ========="
            eval $CMD_T
            #echo "========= ${SCENE}: Render interpolated pose & output video ========="
            #eval $CMD_RI
            done
        done
    done
