
## Build

We have tested building on Ubuntu 18.04 with CUDA 11.8.

```
mkdir build
cd build
cmake ..
make
cd ..
```

```
./build/bin/FaceViewer_FLAME --dataset=DATA_PATH/params --model=MODEL_PATH
```


## Data process

You need to create an account on the [FLAME website](https://flame.is.tue.mpg.de/download.php) and download FLAME 2020 model.
Please unzip FLAME2020.zip and put generic_model.pkl under ./data/FLAME2020.

Then run 
```
python scripts/extract_tensor.py -i ./data/FLAME2020/generic_model.pkl -o ./data/FLAME2020
```

Run code to transform data format for viewer.
```
python scripts/extract_traj.py -i DATA_PATH/checkpoint -o DATA_PATH/params  
```



To reproduce the 370fps performance reported in the paper, please comment out ENABLE_VSYNC at line TODO and uncomment line TODO in `FaceViewer_FLAME.cpp` and rebuild.