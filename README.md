SMaLL-100: Introducing Shallow Multilingual Machine Translation Model for Low-Resource Languages
=================

SMaLL-100 is a compact and fast massively multilingual MT model covering more than 10K language pairs, 
that achieves competitive results with M2M-100 while being much smaller and faster.

We provide the checkpoint in both Fairseq and Huggingface🤗 formats. 

Contents
---------------

- [Fairseq](#fairseq)
- [Huggingface🤗](#huggingface)
- [Tokenization + spBLEU](#tokenize)
- [Citation](#citation)

<a name="fairseq"/>  

Fairseq
--------------  

You should first install the latest version of Fairseq:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
Please follow [fairseq repo](https://github.com/facebookresearch/fairseq) for further detail.

## Generation with SMaLL-100

1. Download pre-trained model from [here]() and put it in ```/model``` directory.
2. Pre-process the evaluation set (sample data is provided in ```/data```).
```
fairseq=/path/to/fairseq
cd $fairseq
for lang in af en ; do
    python scripts/spm_encode.py \
        --model model/spm.128k.model \
        --output_format=piece \
        --inputs=data/test.af-en.${lang} \
        --outputs=spm.af-en.${lang}
done

fairseq-preprocess \
    --source-lang af --target-lang en \
    --testpref spm.af-en \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict model/data_dict.128k.txt --tgtdict model/data_dict.128k.txt
```
3. Translate the data by passing the pre-processed input.
```
fairseq-generate \
   data_bin \
   --batch-size 1 \
   --path model/model.pt \
   --fixed-dictionary model/model_dict.128k.txt \
   -s af -t en \
   --remove-bpe 'sentencepiece' \
   --beam 5 \
   --task translation_multi_simple_epoch \
   --lang-pairs model/language_pairs_small_models.txt \
   --encoder-langtok tgt \
   --gen-subset test > test.af-en.out
 
 cat test.af-en.out | grep -P "^H" | sort -V | cut -f 3- > test.af-en.out.clean
 
 ```
 
<a name="huggingface"/>  

HuggingFace🤗
-----------------  
TODO


<a name="tokenize"/>  

Tokenization + spBLEU
-------------

As mentioned in the paper, we use spBLEU as the MT metric. It uses SentencePiece (SPM) tokenizer with 256K tokens, 
then BLEU is calculated on the tokenized text.
```
git clone --single-branch --branch adding_spm_tokenized_bleu https://github.com/ngoyal2707/sacrebleu.git
cd sacrebleu
python setup.py install
```
To get the score, run:
```
sacrebleu test.af-en.out.ref < test.af-en.out.clean --tokenize spm
```


<a name="citation"/>  

Citation
-------------

<a name="citations"/>  

If you use this code for your research, please cite the following work:
```
@article{mohammadshahi2022small,
  title={SMaLL-100: Introducing Shallow Multilingual Machine Translation Model for Low-Resource Languages},
  author={Mohammadshahi, Alireza and Nikoulina, Vassilina and Berard, Alexandre and Brun, Caroline and Henderson, James and Besacier, Laurent},
  journal={arXiv preprint arXiv:2210.11621},
  year={2022}
}
```
Have a question not listed here? Open [a GitHub Issue](https://github.com/idiap/g2g-transformer/issues) or 
send us an [email](alireza.mohammadshahi@idiap.ch).
