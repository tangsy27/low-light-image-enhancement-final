这份代码需要 libpng 和 fftw3


Linux 服务器：

sudo apt-get update
sudo apt-get install -y build-essential libpng-dev libfftw3-dev
make

编译成功后你会得到 ./MSR_original。


它会生成一个命令行程序 MSR_original，输入一张图，输出两张图：

MSR_rgb：在 RGB 通道上做 multiscale retinex（更像 MSRCR / 对颜色平衡更“激进”）

MSR_gray：在灰度/强度上做 multiscale retinex（更偏向强调局部对比、颜色相对更保留
