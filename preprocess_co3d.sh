python preprocess_co3d.py +arch=votri +dataset=$1 dataset_name=train
python preprocess_co3d.py +arch=votri +dataset=$1 dataset_name=val
python preprocess_co3d.py +arch=votri +dataset=$1 dataset_name=test

