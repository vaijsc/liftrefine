cd $CODE/NVS

CATEGORY=teddybear
EXP_NAME=upsample_vsd
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
            data.no_imgs_per_example=4"


NAME="${DATASET_TYPE}_${ARCH}_${EXP_NAME}"
PRETRAINED_PATH=exp_zero123/co3d_teddybear_votri_zero123_clip/checkpoint/last.pt

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    train_upsample_vsd.py +experiment=diffusion_128 +arch=$ARCH +dataset=$DATASET_TYPE\
                                    logdir=exp_debug name=$NAME batch_size=1 \
                                    num_steps=300010  $CONFIG_STR optimization.classifier_free_guidance=5.0\
                                    pretrained_reconstructor=$PRETRAINED_PATH \
