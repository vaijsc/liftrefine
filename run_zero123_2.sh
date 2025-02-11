cd $CODE/NVS

CATEGORY=teddybear
EXP_NAME=zero123_tune_2
DATA_DIR=$DATA/CO3D_viewset/$CATEGORY/test_object_black_bg
TEST_DIR=$EVAL/phase3
TEST_EMA=true

GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

DATASET_TYPE=co3d_$CATEGORY
ARCH=votri
CONFIG_STR="data.white_background=false
            data.no_imgs_per_example=4
            model.use_depth=false"

LR_STR="lr=0.00001"
NAME="${DATASET_TYPE}_${ARCH}_${EXP_NAME}"
PRETRAINED_PATH=exp_zero123/co3d_teddybear_votri_zero123_clip/checkpoint/best_psnr.pt

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    train_diffusion_stage2.py +experiment=diffusion_128 +arch=$ARCH +dataset=$DATASET_TYPE\
                                    logdir=exp_zero123_2 name=$NAME batch_size=32 \
                                    num_steps=100010 $CONFIG_STR $LR_STR\
                                    pretrained_reconstructor=$PRETRAINED_PATH eval_every=10000 test_steps=50\
                                    #  checkpoint_path=$CHECKPOINT_PATH resume=true