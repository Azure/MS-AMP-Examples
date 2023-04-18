# This is an example of Swin-Transformer using MS-AMP
This example demonstrates how to use MS-AMP in [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

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

## Install dependencies
```
cd Swin-Transformer
pip install -r requirements.txt
```

## Apply patch to Swin-Transformer
We made a few changes to the official Swin-Transformer and packaged it into a patch. You need to apply this patch to third_party/Swin-Transformer.
```
cd ../third_party/Swin-Transformer
git apply ../../Swin-Transformer/Swin-Transformer.patch
cd -
```

## Train Swin-Transformer tiny model with AMP
Run the following command to train a tiny Swin-Transformer model using AMP.
```
sh run.sh tiny amp
```

## Train Swin-Transformer tiny model with MS-AMP
Run the following command to train a tiny Swin-Transformer model using MS-AMP.
```
sh run.sh tiny msamp
```

## Train Swin-Transformer giant model with AMP
Run the following command to train a giant Swin-Transformer model using AMP.
```
sh run.sh giant amp
```

## Train Swin-Transformer giant model with MS-AMP
Run the following command to train a giant Swin-Transformer model using MS-AMP. You can observe significant GPU memory saving using `nvidia-smi` compared with AMP.
```
sh run.sh giant msamp
```