# This is an example of GPT3 using MS-AMP

## Apply patch to Megatron-DeepSpeed
We made a few changes to the official Megatron-DeepSpeed and packaged it into a patch. You need to apply this patch to third_party/Megatron-DeepSpeed.
```
cd ../third_party/Megatron-LM
git apply ../../gpt3/Megatron-LM.patch
cd ../../gpt3
```

## Data preparation
Currently we haven't published the data we use in this example. But we provide a script of preprocessing Wikipedia data from scatch. You can also also use your own data.

```bash
bash prepare_wikipedia.sh
```
After running the above command, a folder named data will be generated. The file structure should look like:
```
$ tree data/
data
├── gpt2-merges.txt
├── gpt2-vocab.json
├── wikipedia_text_document.bin
└── wikipedia_text_document.idx
```
## Pretrain GPT3-345m with fp16
Run the following command to train 345M GPT3 using fp16:
```bash
bash pretrain_345m.sh fp16
```

## Pretrain GPT3-345m with MS-AMP
Run the following command to train 345M GPT3 using MS-AMP:
```
bash pretrain_345m.sh msamp
```
## Pretrain GPT3-13b with bf16
Run the following command to train 13B GPT3 using bf16:
```bash
bash pretrain_13b.sh bf16
```

## Pretrain GPT3-13b with MS-AMP
Run the following command to train 13B GPT3 using MS-AMP:
```bash
bash pretrain_13b.sh msamp
```
