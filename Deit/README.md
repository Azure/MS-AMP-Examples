# This is an example of Deit using MS-AMP
This example, adapted from [Deit]((https://github.com/facebookresearch/deit)), demonstrates how to use ms-amp in a comprehensive training scenario.

## Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision `datasets.ImageFolder`, and the training and validation data is expected to be in the train/ folder and val folder respectively:
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
pip install -r requirements.txt
```

## Run Deit with amp
```
sh run.sh
```

## Run Deit with msamp
```
sh run_msamp.sh
```