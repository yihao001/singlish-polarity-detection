"""
Some code modified from senticnet https://sentic.net/

Go through combined_strict.txt line by line (trad chin script)

For each line, 

ENGLISH 
1. Put the line through Vader (it'll ignore chinese)

Keep only English (and ,.), [remove chinese char]
1. Put whole sentence through vader
2. Split by space and put it through senticnet (word by word), 
3. Check for both English and Singlish via senticnet

CHINESE

Keep only Chinese (and ,.) + remove all spaces  
1. Put through jiagu
2. Check for Chinese via senticnet

Negation detection is perform for both 

To combine, method varies for language due to the idiosyncrasies of each labelling approach
This is further elaborated in the comments below. 

"""

import os
import re

import nltk
from nltk.util import ngrams
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import jiagu
import jieba
list(jieba.cut('不可能')) # warmup

FILENAME = 'combined_strict.txt' 

# Set threshold for +ve and -ve labels, higher will remove noise but lower signal
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1

## Code for negation detection
negation_words = [
    'no', 'not', 'doesnt', 'didnt', 'dont', 'cannot', 'havent', 'wont', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'hadnt', 'wouldnt', 'couldnt', 'cant', 'shouldnt', # Not-negators
    'barely', 'few', 'hardly', 'little', 'neither', 'never', 'nobody', 'none', 'nor', 'nothing', 'nowhere', 'rarely', 'scarcely', 'seldom', # N-negators
    '不', '没' # chinese negators
]

def changeToNo(index, sentence):
    if index < len(sentence):
        replaceChecks = sentence[index] == "and" or sentence[index] == "or" 

        if replaceChecks:
            sentence[index] == "no1"

def checkButYet(index, sentence):
    if (index+1 < len(sentence)):

        if (sentence[index] == "but" or sentence[index] == "yet"):

            wordBefore = sentence[index-1]
            wordAfter = sentence[index+1]
            return True

        return False

    else:
        return False

def negateWord(index, sentence):
    if (index < len(sentence)):
        replaceChecks = (
            (sentence[index] != "and") and
            (sentence[index] != "or") and
            (sentence[index] != "no") and
            (sentence[index] != ",") and
            (sentence[index] != ".")
        )
 
        if replaceChecks:
            thisWord = sentence[index]

            if (thisWord[:2] != "no"):
                sentence[index] = "no1_" + thisWord

def appendNo(word, index, sentence):
    if (word[:2] != "no" and word[:3] != "but"):
        word = "no1_" + word
    return word

def checkNegation(index, sentence):
    """
    Input: Sentence, Output: Whether negated 
    """
    flag = False

    # first word can't be negated, skip it 
    if (index == 0): return False

    # look back previous 3 words 
    previousWord_1 = sentence[index-1]
    if previousWord_1 in negation_words:
        flag = True

    if index > 1:
        previousWord_2 = sentence[index-2]
        if previousWord_2 in negation_words:
            flag = True

    if index > 2:
        previousWord_3 = sentence[index-3]
        if previousWord_3 in negation_words:
            flag = True

    # if negation present, process text further
    if flag:
        if checkButYet(index + 1, sentence):
            return flag

        negateWord(index + 1, sentence)
        changeToNo(index + 1, sentence)

        if checkButYet(index + 2, sentence):
            return flag

        negateWord(index + 2, sentence)
        changeToNo(index + 2, sentence)

        if checkButYet(index + 3, sentence):
            return flag

        negateWord(index + 3, sentence)
        changeToNo(index + 3, sentence)

        if checkButYet(index + 4, sentence):
            return flag

        negateWord(index + 4, sentence)
        changeToNo(index + 4, sentence)

    return flag

def check_negation(sentence):
    negated_words = []
    for i, word in enumerate(sentence):
        if (i > 0 and checkNegation(i, sentence)):
            word = appendNo(word, i, list)
        negated_words.append(word)
    
    return negated_words


def calc_vader_senti(sentence):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    return ss['compound']

import importlib

class SenticNet(object):
    """
    Simple API to use SenticNet 5.

    For more information, refer to:
    E Cambria, Y Li, F Xing, S Poria, K Kwok. SenticNet 6: Ensemble application of symbolic and subsymbolic AI for sentiment analysis. In: CIKM, 105-114 (2020)
    """
    def __init__(self, language="en"):
        data_module = importlib.import_module("senti.senticnet_" + language)
        self.data = data_module.senticnet

    # public methods

    def concept(self, concept):
        """
        Return all the information about a concept: semantics,
        sentics and polarity.
        """
        result = {}

        result["polarity_value"] = self.polarity_value(concept)
        result["polarity_intense"] = self.polarity_intense(concept)
        result["moodtags"] = self.moodtags(concept)
        result["sentics"] = self.sentics(concept)
        result["semantics"] = self.semantics(concept)

        return result

    def semantics(self, concept):
        """
        Return the semantics associated with a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[8:]

    def sentics(self, concept):
        """
        Return sentics of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        sentics = {"pleasantness": concept_info[0],
                   "attention": concept_info[1],
                   "sensitivity": concept_info[2],
                   "aptitude": concept_info[3]}

        return sentics

    def polarity_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[6]

    def polarity_intense(self, concept):
        """
        Return the polarity intense of a concept.
        """

        concept_info = self.data[concept]
        if LANGUAGE == 'sg' or LANGUAGE == 'en':
            return float(concept_info[7])
        elif LANGUAGE == 'my' or LANGUAGE == 'cn':
            return float(concept_info[6])

    def moodtags(self, concept):
        """
        Return the moodtags of a concept.
        """
        concept = concept.replace(" ", "_")
        concept_info = self.data[concept]

        return concept_info[4:6]

sn_en = SenticNet('en')
sn_sg = SenticNet('sg')
# sn_my = SenticNet('my')
sn_cn = SenticNet('cn')

sentiment_scores = {
    'words' : [],
    'vader' : [],
    'sentic_en': [],
    'sentic_sg': [],
    'sentic_ensg': [],
    'sentic_cn': [],
    'jiagu' : [],
}

with open('./data/' + FILENAME) as f:
    for i, line in enumerate(f):
        sentiment_scores['words'].append(line.strip())

        # Vader
        score_vader = calc_vader_senti(line)
        sentiment_scores['vader'].append(score_vader)


        # SenticNet (en, sg)
        cleaned_line = re.sub(r'[\u4e00-\u9fff]+', '', line) ## Remove Chinese 
        cleaned_line = re.sub('-', ' ', cleaned_line)
        cleaned_line = re.sub('[^\w,. ]', '', cleaned_line) # remove non Az.,
        cleaned_line = re.sub('_', '', cleaned_line) # remove _ since in \w
        cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line) # remove double space
        cleaned_line = cleaned_line.strip()
        
        score_sentic_en = 0
        score_sentic_sg = 0
        score_sentic_ensg = 0
        count_en = 0
        count_sg = 0

        cleaned_line_list = cleaned_line.split(' ')
        negated_line_list = check_negation(cleaned_line_list)

        for word, negated_word in zip(cleaned_line_list, negated_line_list):
            try:
                LANGUAGE = 'en' # en cn my sg
                en_polarity_score = sn_en.polarity_intense(word.lower())
                if 'no1' in negated_word:
                    score_sentic_en += -en_polarity_score
                else:
                    score_sentic_en += en_polarity_score
                count_en += 1
            except KeyError:
                try: 
                    LANGUAGE = 'sg' # en cn my sg
                    sg_polarity_score = sn_sg.polarity_intense(word.lower())
                    if 'no1' in negated_word:
                        score_sentic_sg += -sg_polarity_score 
                    else:
                        score_sentic_sg += sg_polarity_score 
                    count_sg += 1
                except KeyError:
                    continue
        
        for bigram, negated_bigram in zip(ngrams(cleaned_line_list,2), ngrams(negated_line_list,2)):
            word = ' '.join(bigram)
            negated_word = ' '.join(negated_bigram)
            try:
                LANGUAGE = 'en' # en cn my sg
                en_polarity_score = sn_en.polarity_intense(word.lower())
                if 'no1' in negated_word:
                    score_sentic_en += -en_polarity_score
                else:
                    score_sentic_en += en_polarity_score
                count_en += 1
            except KeyError:
                try: 
                    LANGUAGE = 'sg' # en cn my sg
                    sg_polarity_score = sn_sg.polarity_intense(word.lower())
                    if 'no1' in negated_word:
                        score_sentic_sg += -sg_polarity_score 
                    else:
                        score_sentic_sg += sg_polarity_score 
                    count_sg += 1
                except KeyError:
                    continue

        for trigram, negated_trigram in zip(ngrams(cleaned_line_list,3), ngrams(negated_line_list,3)):
            word = ' '.join(trigram)
            negated_word = ' '.join(negated_trigram)
            try:
                LANGUAGE = 'en' # en cn my sg
                en_polarity_score = sn_en.polarity_intense(word.lower())
                if 'no1' in negated_word:
                    score_sentic_en += -en_polarity_score
                else:
                    score_sentic_en += en_polarity_score
                count_en += 1
            except KeyError:
                try: 
                    LANGUAGE = 'sg' # en cn my sg
                    sg_polarity_score = sn_sg.polarity_intense(word.lower())
                    if 'no1' in negated_word:
                        score_sentic_sg += -sg_polarity_score 
                    else:
                        score_sentic_sg += sg_polarity_score 
                    count_sg += 1
                except KeyError:
                    continue

        if (not count_en == 0) or (not count_sg == 0):
            score_sentic_ensg = (score_sentic_en + score_sentic_sg) / (count_en + count_sg)

        if not count_en == 0: # else case will be 0 so okay
            score_sentic_en = score_sentic_en/count_en

        if not count_sg == 0:
            score_sentic_sg = score_sentic_sg/count_sg
            
        sentiment_scores['sentic_en'].append((score_sentic_en, count_en))
        sentiment_scores['sentic_sg'].append((score_sentic_sg, count_sg))
        sentiment_scores['sentic_ensg'].append((score_sentic_ensg, count_en + count_sg))


        # Chinese
        # http://mylanguages.org/chinese_negation.php
        # Doesn't need it as jieba will do it automatically 
        cleaned_line = re.sub(r'[^\u4e00-\u9fff]+', '', line) # keep only chinese
        cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line) # remove multiple spaces

        score_cn_jiagu_raw = jiagu.sentiment(cleaned_line)
        if score_cn_jiagu_raw[0] == 'positive':
            score_cn_jiagu = ((score_cn_jiagu_raw[1] - 0.5) / 0.5)
        else: 
            score_cn_jiagu = -((score_cn_jiagu_raw[1] - 0.5) / 0.5)

        sentiment_scores['jiagu'].append(score_cn_jiagu)

        
        ## Sentic_cn

        score_sentic_cn = 0
        count_words_with_sentiment = 0
        count_cn = 0 

        chinese_parts = list(jieba.cut(cleaned_line))
        negated_line_list = check_negation(chinese_parts)
            
        for word, negated_word in zip(chinese_parts, negated_line_list):
            try:
                LANGUAGE = 'cn' # en cn my sg
                cn_polarity_score = sn_cn.polarity_intense(word.lower())
                if 'no1' in negated_word:
                    score_sentic_cn += -cn_polarity_score
                else:
                    score_sentic_cn += cn_polarity_score
                count_cn += 1
            except KeyError:
                continue
        
        for bigram, negated_bigram in zip(ngrams(chinese_parts,2), ngrams(negated_line_list,2)):
            word = ''.join(bigram)
            negated_word = ''.join(negated_bigram)
            try:
                cn_polarity_score = sn_cn.polarity_intense(word.lower())
                if 'no1' in negated_word:
                    score_sentic_cn += -cn_polarity_score
                else:
                    score_sentic_cn += cn_polarity_score
                count_cn += 1
            except KeyError:
                continue

        for trigram, negated_trigram in zip(ngrams(chinese_parts,3), ngrams(negated_line_list,3)):
            word = ''.join(trigram)
            negated_word = ''.join(negated_trigram)
            try:
                cn_polarity_score = sn_cn.polarity_intense(word.lower())
                if 'no1' in negated_word:
                    score_sentic_cn += -cn_polarity_score
                else:
                    score_sentic_cn += cn_polarity_score
                count_cn += 1
            except KeyError:
                continue
        
        if not count_cn == 0: # else case will be 0 so okay
            score_sentic_cn = score_sentic_cn/count_cn

        sentiment_scores['sentic_cn'].append((score_sentic_cn, count_cn))

        if i % 10000 == 1000:
            print(i)

sentiment_scores['sum_en'] = []
sentiment_scores['sum_cn'] = []
sentiment_scores['total'] = []
sentiment_scores['label'] = []

for words, vader, (en, count_en), (sg, count_sg), (ensg, count_ensg), (cn, count_cn), jiagu in zip(sentiment_scores['words'], sentiment_scores['vader'], sentiment_scores['sentic_en'], sentiment_scores['sentic_sg'], sentiment_scores['sentic_ensg'], sentiment_scores['sentic_cn'], sentiment_scores['jiagu']):

    # English
    # Vader is good but doesn't catch Singlish. en catches concepts... but it's noisy esp if short
    # sg catches Singlish. 
    if vader == 0.0:
        if ensg == 0.0:
            english_score = 0.0 # both neutral / not in vocab
        else: # en has value but not vader. might be due to sg or en 
            english_score = ensg if count_ensg > 2 else 0.0 # if en too short, might be eccentric
    else:
        if ensg == 0.0:
            english_score = vader # might not be in ensg, so follow vader
        else: # both not 0 
            if vader * ensg > 0.0: # both positive or negative
                english_score = (vader + ensg)/2
            else: # different sign, we trust singlish more if available
                if count_sg > 1 or count_ensg / len(words) > 0.3:
                    english_score = (vader + ensg)/2
                else:
                    english_score = vader
                                
    # Chinese (lower chance of not being in vocab, vs English)
    # sentic_cn is very small. jiagu is more reliable but not concept-based (besides negation)
    if cn == 0.0:
        if jiagu == 0.0:
            chinese_score = 0.0 # if both are 0, it's probably 0 
        else:
            chinese_score = jiagu # prolly cn missed out due to small vocab
    else: # cn isn't 0!  only keep if strong enough
        if jiagu == 0.0: # cn might catch words with connotations that jiagu missed
            chinese_score = 0.0 if count_cn < 2 else cn # only 1 might be too noisy
        else: # both not 0 
            if cn * jiagu > 0.0: # if they have same direction
                chinese_score = (cn + jiagu)/2 if count_cn > 1 else jiagu # take their average
            else: # if disagree, trust jiagu more if count_cn small 
                chinese_score = jiagu if count_cn < 3 else (cn + jiagu)/2

    sentiment_scores['sum_en'].append(english_score)
    sentiment_scores['sum_cn'].append(chinese_score)

    # since en and cn might have different sentiments, should share equally regardless of length
    total_score = english_score/2 + chinese_score/2
    sentiment_scores['total'].append(total_score)

    label = 'positive' if total_score > POSITIVE_THRESHOLD else 'negative' if total_score < NEGATIVE_THRESHOLD else 'neutral'

    sentiment_scores['label'].append(label)

# Save dictionary

TARGET_DIRECTORY = './data/' + FILENAME[:-4] + '/'
if not os.path.isdir(TARGET_DIRECTORY):
    os.makedirs(TARGET_DIRECTORY)

import csv
w = csv.writer(open(TARGET_DIRECTORY + "scores_raw.csv", "w", newline='', encoding='utf-8-sig'))
for key, val in sentiment_scores.items():
    w.writerow([key, val])

import pickle
f = open(TARGET_DIRECTORY + "scores.pkl","wb")
pickle.dump(sentiment_scores,f)
f.close()

with open(TARGET_DIRECTORY + "scores.csv", 'w', newline='', encoding='utf-8-sig') as myfile:
    wr = csv.writer(myfile)
    for key, value in sentiment_scores.items():
        wr.writerow(value)

with open(TARGET_DIRECTORY + "labels.txt", 'w') as myfile:
    wr = csv.writer(myfile)
    for l in sentiment_scores['label']:
        wr.writerow([l])

with open(TARGET_DIRECTORY + "labels.csv", 'w', newline='',encoding='utf-8-sig') as myfile:
    wr = csv.writer(myfile)
    for w,l in zip(sentiment_scores['words'],sentiment_scores['label']):
        wr.writerow([w,l])
