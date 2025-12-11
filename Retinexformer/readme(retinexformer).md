1. Create environment
```bash
conda create -n Retinexformer python=3.9
conda activate Retinexformer
pip install torch torchvision
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

2. Install BasicSR (CPU mode):
```bash
python setup.py develop --no_cuda_ext
```

3. Test on LOL-v1:

```bash
python Enhancement/test_from_dataset.py \
    --opt Options/RetinexFormer_LOL_v1.yml \
    --weights pretrained_weights/LOL_v1.pth \
    --dataset LOL_v1
```
