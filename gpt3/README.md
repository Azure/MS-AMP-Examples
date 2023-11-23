# This is an example of GPT3 using MS-AMP
We support both of Megatron-DeepSpeed and Megatron-LM. You can choose either of them to run GPT-3.

## Install dependencies
You need to install depedencies before training GPT3. It is recommended to use venv for virtual environments, but it is not strictly necessary.
```bash
pip install einops nltk wikiextractor
```

## Data preparation
Currently we haven't published the data we use in this example. But we provide a script of preprocessing Wikipedia data from scatch. Make sure you have more than 40GB space on your disk and it may take ~4 hours. You can also also use your own data.

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

### Pretrain GPT3-345m
Run the following command to train 345M GPT3 using bf16, Transformer-Engine and MS-AMP:
```bash
bash pretrain_345m_megatron.sh bf16
bash pretrain_345m_megatron.sh te
bash pretrain_345m_megatron.sh msamp
```

Please note that currently MS-AMP may not outperform Transformer-Engine for small models.

### Pretrain GPT3-6.7b
Run the following command to train 6.7B GPT3 using bf16, Transformer-Engine and MS-AMP:
```bash
bash pretrain_6b7_megatron.sh bf16
bash pretrain_6b7_megatron.sh te
bash pretrain_6b7_megatron.sh msamp
```

### Pretrain GPT3-13b
Run the following command to train 13B GPT3 using bf16, Transformer-Engine and MS-AMP:
```bash
bash pretrain_13b_megatron.sh bf16
bash pretrain_13b_megatron.sh te
bash pretrain_13b_megatron msamp
```
You may get out-of-memory error when using Tranformer-Engine since Transformer-Engine consumes more memory than bf16 and MS-AMP. 

## Using Megatron-DeepSpeed

### Apply patch to Megatron-DeepSpeed
We made a few changes to the official Megatron-DeepSpeed and packaged it into a patch. You need to apply this patch to third_party/Megatron-LM.
```bash
cd ../third_party/Megatron-DeepSpeed
git apply ../../gpt3/Megatron-DeepSpeed.patch
cd ../../gpt3
```

### Pretrain GPT3-345m
Run the following command to train 345M GPT3 using fp16 and MS-AMP:
```bash
bash pretrain_345m_megatron_ds.sh fp16
bash pretrain_345m_megatron_ds.sh msamp
```

### Pretrain GPT3-13b

Run the following command to train 13B GPT3 using bf16 and MS-AMP:
```bash
bash pretrain_13b_megatron_ds.sh bf16
bash pretrain_13b_megatron_ds.sh msamp
```

## Multi-node training
If you want to train GPT-3 with Megatron-LM using multiple nodes, you need:
- Upload data to a shared storage and mount the shared storage to each node.
- Change MASTER_ADDR, NNODES, NODE_RANK in the script.
- [optional] Set some environment variables related to RDMA before running the script. For example, if you are using [ND H100 v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series), you need to set these environment variables:
  ```bash
  export NCCL_IB_PCI_RELAXED_ORDERING=1
  export NCCL_SOCKET_IFNAME=eth0
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export NCCL_NET_GDR_LEVEL=5
  export NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml
  export NCCL_DEBUG=WARN
  ```
- Use a parallel ssh tool to start the script in all nodes.
