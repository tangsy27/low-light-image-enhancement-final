1. Create conda environment
```bash
conda create -n retinexnet python=3.7 -y
conda activate retinexnet
```
2. Install dependencies

```bash
pip install "tensorflow==1.15"
pip install numpy pillow
```
3. Download code & prepare checkpoints
   
4. Copy the existing Decom/ and Relight/ weight folders into the checkpoint/ directory at the project root.

5. we need to put the input image into a project like:

```bash
root/projects/lowlight_test/
```

4. run tesing (CPU)
```bash
python main.py \
    --use_gpu=0 \
    --phase=test \
    --test_dir=/root/projects/lowlight_test/ \
    --save_dir=/root/projects/lowlight_out/ \
    --decom=0
```
