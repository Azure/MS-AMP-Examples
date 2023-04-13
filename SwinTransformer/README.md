# This is an example of SwinTransformer using MS-AMP
This example, adapted from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), demonstrates how to use ms-amp in a comprehensive training scenario.

## Install dependencies
```
pip install -r requirements.txt
```

## Data preparation
We use standard ImageNet dataset, you can download it from http://image-net.org/. The file structure should look like:
```
$ tree data
ImageNet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```
After that, you may need to change the varaible DATA_PATH to the data folder in launch script.

## Run swin-transformer with amp
```
sh run.sh
```

## Run swin-transformer with msamp
```
sh run_msamp.sh
```