# Intelligent System Final
## Contents
```
- KalmanFilter.py     A Kalman filter 
- MothionAnalyser.py  Analyze the motion of frames
- NCC.py              GPU accelerated NCC algorithm
- Stabliser.py        Entry of the whole toolkit
- utils.py            Supporting functions
- Videoloader.py      Load the video
- VideoProcessor.py   Output the video, perform translation
```
## Usage
Unfortunately, there is only the GPU version of the NCC algorithm... As a result you must have an Nvidia  graphic card and a properly configured CUDA toolkit to run the code. The `CuPy` package is also obligatory,

The script is tested on a host with Ubuntu 18.04 LTS, with an Intel i7 processor, an RTX 2060 Super graphic card and 16GB RAM.

### Steps
1. Install the `CuPy` with instructions from [Installation Guide](https://docs-cupy.chainer.org/en/stable/install.html). The `CuPy` package names are different depending on the CUDA version you have installed on your host. For CUDA 10.1, execute `pip install cupy-cuda101`.
2. Install the OpenCV by `pip install opencv-python` and `pip install opencv-python-contrib`
3. Install other necessary packages
4. Run Stabliser.py with parameters, for example
```bash
python Stabliser.py --path_to_video ./videos/IMG_1442.MOV
```
5. The output video is saved to `./output/` by default