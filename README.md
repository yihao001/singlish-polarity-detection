# Modelling code-switching in Singlish for polarity detection

[Project page](https://yihao001.github.io/projects/ce7455_singlish/)

A subset of the most relevant preprocessed data and code (that should be most relevant for further work in this area) is available in this repository. If there are any missing files, do create a GitHub issue and I should be able to provide them. 

## Datasets

1. [National Speech Corpus](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus): around 1 million lines stored in the `TextGrid` format, from conversations with code-mixing
2. [NUS SMS Corpus](https://github.com/kite1988/nus-sms-corpus): around 30k English and 30k Chinese SMS messages from Singaporean university students
3. [SEAME](https://github.com/zengzp0912/SEAME-dev-set/): around 10,000 lines of text with code-mixing (a mix of English and Chinese dominant lines)
4. [Singapore Bilingual Corpus](https://childes.talkbank.org/access/Biling/Singapore.html): around 50,000 lines stored in the `CHILDES` format and parsed by the `PyLangAcq` library
5. [Malaya dataset](https://github.com/mesolitica/malaysian-dataset): around 19 million sentences crawled from local online forums

Raw datasets can be obtained from the above links. The processed dataset (combined from most/all sources) is available in the `data` folder of this repository.

## Data processing

The datasets above have very different formatting, so that has to be made consistent (i.e. remove tags or any dataset-specific annotations). This is done in `dataprep.py` using a dollop of regular expressions and various language and filetype-specific libraries. 
- In general, there is a dedicated function for cleaning each dataset. 
- Then, `prep_combined()` combines these datasets and generates various forms of the combined dataset (English only, English + Pinyin, Chinese only, Mixed). 
- While the function creates 2 forms of dataset ('strict', where only utterances with both English and Chinese characters are kept; and a less strict version as description in the report that leads to a bigger but less clean dataset), only the 'strict' version was used due to a lack of compute. 

With a consistent dataset, it is then feasible to create labels. `datalabel.py` does so via various sentiment analysis libraries (NLTK Vader, SenticNet and Jiagu) and a multi-lingual negation detection algorithm. It requires some files from the [SenticNet repo](https://github.com/yurimalheiros/senticnetapi), such as `senticnet_cn.py`.

The labels depend on a set of thresholds that can be adjusted based on a subset of manually labelled data. This is adjusted in `labels_kappa.csv`. It requires the creation of a `labels_kappa.csv` file which needs 3 columns: the text snippet, the predicted label and the manual label ('ground truth').

## Models

Checkpoints for the pre-trained models used (`uncased_L-12_H-768_A-12`, `chinese_L-12_H-768_A-12`, `multi_cased_L-12_H-768_A-12`) can be obtained [here](https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md#checkpoints).

In total, 8 experiments were conducted. They are described below, along with the name of the Python file that implements them. 

1. Baseline model (English): `model_baseline_enonly.py`
2. Baseline model (English + Pinyin): `model_baseline_en.py`
3. Baseline model (Chinese only): `model_baseline_cn.py`
4. Baseline model (Chinese only, segmented with Jieba): `model_baseline_cn_seg.py`
5. Baseline model (Mixed): `model_baseline_mul.py`
6. Pre-trained model (English + Pinyin): `model_pretrained_en.py`
7. Pre-trained model (Chinese only): `model_pretrained_cn.py`
8. Pre-trained model (Mixed): `model_pretrained_mul.py`

Code for the best performing model (Pre-trained, Mixed) is made available in this repository.

