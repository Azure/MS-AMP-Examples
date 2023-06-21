#!/bin/bash
set -x

mkdir -p data
cd data

echo "start to download Wikipedia dump"
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

echo "download completed, start to extract json files"
pip install wikiextractor
python -m wikiextractor.WikiExtractor --json enwiki-latest-pages-articles.xml.bz2

echo "extract completed, start to merge json files"
ouput_json="wiki_all.json"
find text/ -name wiki* | parallel -m -j 70 "cat {} >> ${ouput_json}"

echo "merge completed, start to preprocess"
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
python ../../third_party/Megatron-DeepSpeed/tools/preprocess_data.py \
       --input $ouput_json \
       --output-prefix wikipedia \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod \
       --workers 70

echo "preprocess completed, start to remove temporary files"

rm -rf enwiki-latest-pages-articles.xml.bz2
rm -rf text/
rm -rf $ouput_json

cd ../
