EXP_NAME=generate_fid_stat

python generate_fid_stat.py \
                        +arch=votri +dataset=co3d_teddybear name=$EXP_NAME data.white_background=true


python generate_fid_stat.py \
                        +arch=votri +dataset=co3d_teddybear name=$EXP_NAME data.white_background=false

