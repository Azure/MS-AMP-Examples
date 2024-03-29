# MS-AMP Examples
This repository contains various training examples including [DeiT](https://github.com/facebookresearch/deit), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [RoBERTa](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md) and [GPT-3](https://github.com/microsoft/Megatron-DeepSpeed#gpt-pretraining) that use [MS-AMP](https://github.com/Azure/MS-AMP).

# Get started

## Prerequisites
In order to run examples in this repository, you need to install MS-AMP first, and  then clone the repository and submodule with the following command:
```
git clone https://github.com/Azure/MS-AMP-Examples.git
cd MS-AMP-Examples
git submodule update --init --recursive
```

## Swin-Transformer
This folder contains end-to-end training of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) model using MS-AMP.

## DeiT
This folder contains end-to-end training of [DeiT](https://github.com/facebookresearch/deit) model using MS-AMP.

## RoBERTa
This folder contains end-to-end training of [RoBERTa](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md) model using MS-AMP.

## GPT-3
This folder contains end-to-end training of [GPT-3](https://github.com/NVIDIA/Megatron-LM#gpt-pretraining) model using MS-AMP.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
