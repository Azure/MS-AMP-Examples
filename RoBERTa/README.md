# This is an example of RoBERTa using MS-AMP
This example demonstrates how to MS-AMP in [RoBERTa](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md).

## Data preparation
Currently we haven't published the data we use in this example. You can use public dataset such as [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) or your own data. Please see the [tutorial for pretraining RoBERTa using your own data](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md).

Here is an example of preparing WikiText-103 dataset:
```
sh prepare_wikitext.sh
```
After running the above command, a folder named data-bin will be generated. The file structure should look like:
```
$ tree data-bin/
data-bin/
└── wikitext-103
    ├── dict.txt
    ├── preprocess.log
    ├── test.bin
    ├── test.idx
    ├── train.bin
    ├── train.idx
    ├── valid.bin
    └── valid.idx
```

If you use your own data, don't forget to change the variable DATA_PATH to the data folder in launch script.

## Apply patch to fairseq
We made a few changes to the official [fairseq](https://github.com/facebookresearch/fairseq) and packaged it into a patch. You need to apply this patch to third_party/fairseq.
```
cd ../third_party/fairseq
git apply ../../RoBERTa/fairseq.patch
```

## Install failseq
You need to install fairseq before training RoBERTa. It is recommended to use venv for virtual environments, but it is not strictly necessary.
```
pip install --no-build-isolation -v -e .
cd -
```
You can verify if the installation of fairseq by executing `python -c "import fairseq; print(fairseq.__version__)"`.

## Train RoBERTa model with AMP
Run the following command to train RoBERTa base model using AMP:
```
sh run.sh amp
```

## Train RoBERTa model with MS-AMP
Run the following command to train RoBERTa base model using MS-AMP:
```
sh run.sh msamp
```
