cd $CODE/NVS

CATEGORY=hydrant
EXP_NAME=zero123_tune_1
DATA_DIR=$DATA/CO3D_viewset_256/$CATEGORY/test_object_black_bg
TEST_DIR=$EVAL/official/diffusion
TEST_EMA=true


GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

DATASET_TYPE=co3d_$CATEGORY
ARCH=votri
CONFIG_STR="data.white_background=false
            data.no_imgs_per_example=4
            data.novel_view_weight=0.05"

NAME="${DATASET_TYPE}_${ARCH}_${EXP_NAME}"
CHECKPOINT_PATH=exp_official/co3d_hydrant_votri_zero123_tune_1/checkpoint/best_psnr.pt

for NUM_TEST_VIEWS in 1 3
do
echo $TEST_DIR/$DATASET_TYPE/$EXP_NAME/ema_${TEST_EMA}_${NUM_TEST_VIEWS}
CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    evaluation_qualitative_co3d_diffusion.py +arch=$ARCH +dataset=$DATASET_TYPE \
                                    +experiment=diffusion_128 \
                                    logdir=exp_debug name=$NAME batch_size=32 \
                                    num_test_views=$NUM_TEST_VIEWS test_ema=$TEST_EMA $CONFIG_STR \
                                    checkpoint_path=$CHECKPOINT_PATH \
                                    test_dir=$TEST_DIR/$DATASET_TYPE/$EXP_NAME/ema_${TEST_EMA}_${NUM_TEST_VIEWS}

python evaluation/evaluate.py -D $DATA_DIR \
                                -O $TEST_DIR/$DATASET_TYPE/$EXP_NAME/ema_${TEST_EMA}_${NUM_TEST_VIEWS} --gpu_id=$FIRST_GPU

done