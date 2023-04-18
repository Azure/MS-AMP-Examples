# This is an example of DeiT using MS-AMP
This example, adapted from [DeiT](https://github.com/facebookresearch/deit), demonstrates how to use MS-AMP in a comprehensive training scenario.

## Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision `datasets.ImageFolder`, and the training and validation data is expected to be in the train folder and val folder respectively:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
After that, you may need to change the varaible DATA_PATH to the data folder in launch script.

## Install dependencies
```
cd deit
pip install -r requirements.txt
```

## Apply patch to DeiT
We made a few changes to the official DeiT and packaged it into a patch. You need to apply this patch to third_party/deit.
```
cd ../third_party/deit
git apply ../../deit/deit.patch
cd -
```

## Run DeiT-Small with AMP
Run the following command to train a small DeiT model using AMP.
```
sh run.sh
```

## Run DeiT-Small with MS-AMP
Run the following command to train a small DeiT model using MS-AMP.
```
sh run_msamp.sh
```

## Run DeiT-Large with AMP
Run the following command to train a large DeiT model using AMP. The model has 1.3 billion parameters.
```
sh run_large.sh
```
## Run Deit-Large with MS-AMP
Run the following command to train a large DeiT model using MS-AMP. You can observe significant GPU memory saving using `nvidia-smi` compared with AMP.
```
sh run_large_msamp.sh
```