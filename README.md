## Project Page
https://chen3110.github.io/adaptivefusion/index.html

## Install

Our codebase is developed based on [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) and [MeshTransformer](https://github.com/microsoft/MeshTransformer). Please check [Install.md](https://github.com/microsoft/MeshGraphormer/blob/main/docs/INSTALL.md) to install the relevant dependencies. Please also consider citing them if you use this codebase.

## Download

Visit [mmBody](https://github.com/Chen3110/mmBody) to download mmBody dataset and place it at `./datasets`. 

The pre-trained models can be downloaded with the following command.
```bash
sh download_models.sh
```

Visit the following websites to download SMPL and SMPL-X models. 

- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `./src/modeling/data`.
- Download `SMPLX_NEUTRAL.pkl` from [SMPL-X](https://smpl-x.is.tue.mpg.de/), and place it at `./src/modeling/data`.

## Experiments
### Training
We use the following script to train on the mmBody dataset.

```bash
python ./run_adaptivefusion.py \
    --output_dir output/adaptivefusion \
    --dataset mmBodyDataset \
    --data_path datasets/mmBody \
    --mesh_type smplx \
    --inputs image0,image1,depth0,depth1,radar0 \
    --model AdaptiveFusion \
    --per_gpu_train_batch_size 10 \
    --train \
    --eval_test_dataset
```

### Testing

```bash
python ./run_adaptivefusion.py \
    --output_dir output/adaptivefusion \
    --resume_checkpoint output/adaptivefusion/checkpoint \
    --dataset mmBodyDataset \
    --data_path datasets/mmBody \
    --mesh_type smplx \
    --inputs image0,image1,depth0,depth1,radar0 \
    --model AdaptiveFusion \
    --test_scene lab1
```


## BibTeX
```
@article{chen2024adaptivefusion,
  title={AdaptiveFusion: Adaptive Multi-Modal Multi-View Fusion for 3D Human Body Reconstruction},
  author={Chen, Anjun and Wang, Xiangyu and Xu, Zhi and Shi, Kun and Qin, Yan and Huo, Yuchi and Chen, Jiming and Ye, Qi},
  journal={arXiv preprint arXiv:2409.04851},
  year={2024}
}
```
