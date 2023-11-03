# This is an example of GPT3 using MS-AMP
We support both of Megatron-DeepSpeed and Megatron-LM. You can choose either of them to run GPT-3.

## Install dependencies
You need to install depedencies before training GPT3. It is recommended to use venv for virtual environments, but it is not strictly necessary.
```bash
pip install einops nltk wikiextractor
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

## Using Megatron-LM

### Apply patch to Megatron-LM
We made a few changes to the official Megatron-LM and packaged it into a patch. You need to apply this patch to third_party/Megatron-LM.
```bash
cd ../third_party/Megatron-LM
git apply ../../gpt3/Megatron-LM.patch
cd ../../gpt3
```

## Pretrain GPT3-345m with bf16
Run the following command to train 345M GPT3 using bf16:
```bash
bash pretrain_345m_megatron.sh bf16
```

## Pretrain GPT-345m with Transformer-Engine
Run the following command to train 345M GPT3 using Transformer-Engine:
```bash
bash pretrain_345m_megatron.sh te
```

## Pretrain GPT3-345m with MS-AMP
Run the following command to train 345M GPT3 using MS-AMP:
```bash
bash pretrain_345m_megatron.sh msamp
```

## Pretrain GPT3-13b with bf16
Run the following command to train 13B GPT3 using bf16:
```bash
bash pretrain_13b_megatron.sh bf16
```

## Pretrain GPT3-13b with Transformer-Engine
Run the following command to train 13B GPT3 using Transformer-Engine:
```bash
bash pretrain_13b_megatron te
```

## Pretrain GPT3-13b with MS-AMP
Run the following command to train 13B GPT3 using MS-AMP:
```bash
bash pretrain_13b_megatron.sh msamp
```

## Using Megatron-DeepSpeed

### Apply patch to Megatron-DeepSpeed
We made a few changes to the official Megatron-DeepSpeed and packaged it into a patch. You need to apply this patch to third_party/Megatron-LM.
```bash
cd ../third_party/Megatron-DeepSpeed
git apply ../../gpt3/Megatron-DeepSpeed.patch
cd ../../gpt3
```

## Pretrain GPT3-345m with fp16
Run the following command to train 345M GPT3 using fp16:
```bash
bash pretrain_345m_megatron_ds.sh fp16
```

## Pretrain GPT3-345m with MS-AMP
Run the following command to train 345M GPT3 using MS-AMP:
```bash
bash pretrain_345m_megatron_ds.sh msamp
```
## Pretrain GPT3-13b with bf16
Run the following command to train 13B GPT3 using bf16:
```bash
bash pretrain_13b_megatron_ds.sh bf16
```

## Pretrain GPT3-13b with MS-AMP
Run the following command to train 13B GPT3 using MS-AMP:
```bash
bash pretrain_13b_megatron_ds.sh msamp
```
