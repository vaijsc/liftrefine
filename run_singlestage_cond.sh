cd $CODE/NVS

EXP_NAME=baseline_001_act_normfactor
DATA_DIR=$DATA/CO3D_viewset/hydrant/test_object_black_bg
TEST_DIR=$EVAL/phase3
TEST_EMA=true


GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

DATASET_TYPE=co3d_hydrant
NAME="${DATASET_TYPE}_${EXP_NAME}"

ARCH=votri
CONFIG_STR="random_drop_view=true
            optimization.use_rays=false 
            optimization.use_resnet=false 
            optimization.recons_weight=1.0
            optimization.novel_weight=0.1
            optimization.lpips_weight=0.2
            optimization.tv_weight=0.0
            optimization.depth_weight=0.0
            model.use_depth=false 
            model.unet.clipping=false 
            model.unet.use_act=true
            data.white_background=false
            optimization.diffusion_weight=0.001"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    train_recons_diffusion_cond.py +arch=$ARCH +dataset=$DATASET_TYPE \
                                    +experiment=diffusion_128 \
                                    logdir=exp_debug name=$NAME batch_size=2\
                                    $CONFIG_STR eval_every=2 # checkpoint_path=exp_diffusion/$NAME/checkpoint/last.pt
