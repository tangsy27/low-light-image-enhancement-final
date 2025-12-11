>This code requires **libpng** and **fftw3**.

1. Dependencies (Linux): Install on a Linux server:
```bash
sudo apt-get update
sudo apt-get install -y build-essential libpng-dev libfftw3-dev make
```
2. After compiling, you will get an executable:
 ```bash
./MSR_original
```
3. This command-line program takes one input image and produces two outputs:
   
    a. MSR_rgb: Multiscale Retinex applied on RGB channels

    b. MSR_gray: Multiscale Retinex applied on grayscale/intensity
