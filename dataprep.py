"""
In total, there are 5 datasets

1. National Speech Corpus (1000 hours)
2. The National University of Singapore SMS Corpus (~60k messages)
3. Code-switching datasets such as SEAME (10 hours)
4. Singapore Bilingual Corpus (50 hours)
5. Malaya dataset (~19 million messages of scraped data from forums and blogs)

prep_combine() combines these datasets together and performs further cleaning steps

Another function generates a list of Singlish words compiled from multiple web sources

Finally, generate_data() is called by the model_*.py files to generate the training, validation, and test datasets
"""

import os
import re
import numpy as np

from collections import OrderedDict
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def prep_nsc():
    """
    Many transcription markers have to be replaced manually, as seen in `od` below

    Data source: https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus
    TextGrid parser used: https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/textgrid.py
    http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html
    """
    
    from textgrid import TextGrid

    od = OrderedDict(
        [
            ("<FIL/>", ""),
            ("<NEN>", ""),
            ("<nen>", ""),
            ("<NON/>", ""),
            ("<NPS/>", ""),
            ("<SPK/>", ""),
            ("<STA/>", ""),
            
            ("<unk>", ""),
            ("<UNK>", ""),
            ("<UKN>", ""),
            ("<UNK/>", ""),

            ("<c/>", ""),
            ("<s/>", ""),
            ("<S/>", ""), # newly added

            ("**", ""),
            ("  ", " "),
        ])

    CONVENTION_MARKERS_TO_EXCLUDE = ['<Z>','<S>'] # if the line only contains this, skip immediately
    LINE_LENGTH_THRESHOLD = 10 # lines shorter than x char are not saved

    SOURCE_DIRECTORY = './data/1_nsc/'

    conversations = []

    files_with_error_count = 0
    total_file_count = 0
    
    for f in os.listdir(SOURCE_DIRECTORY):
        if '.TextGrid' in f:
            total_file_count +=1 
            # print(f)

            with open(os.path.join(SOURCE_DIRECTORY, f), encoding='utf-16') as script:
                grid = script.read()

            if '\ufeff' in grid: 
                # not sure why this is still a problem even though the encoding used when reading the file should be correct
                # https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in

                grid = grid[1:] # remove extra \ufeff

            try:
                fid = TextGrid(grid)
            except IndexError: # not sure
                print("Error in " + f)
                files_with_error_count += 1
                continue

            for i, tier in enumerate(fid):

                transcript = tier.simple_transcript

                for start_time, end_time, line in transcript:
                    if line not in CONVENTION_MARKERS_TO_EXCLUDE:
                        cleaned_line = replace_all(line, od)
                        if len(cleaned_line) >= LINE_LENGTH_THRESHOLD:
                            conversations.append(cleaned_line.strip())

    print("Total number of files: ", total_file_count)
    print("Number of files with error (skipped): ", files_with_error_count)
    print("Length of dataset: ", len(conversations))

    with open(os.path.join(SOURCE_DIRECTORY, "nsc_conversations.txt"), 'w') as f:
        for line in conversations[:-1]:
            f.write(line + "\n")
        f.write(conversations[-1])

def prep_nussms(language_choice='en'):
    """
    Code modified from https://github.com/jasonyip184/SGTextGenerationLSTM 
    He used the National University of Singapore SMS Corpus for Singlish text generation (but only for English)

    https://www.kaggle.com/sangee1301/analysis-on-english-smsdata
    """
    
    import json
    from statistics import mean, median, mode

    SOURCE_DIRECTORY = './data/2_nus_sms/'
    od = OrderedDict(
        [
            ('<#>',''),
            ('<DECIMAL>',''),
            ('<EMAIL>',''),
            ('<URL>',''),
            ('<TIME>',''),
            ('<DATE>',''),
            ('<name>',''),
            ('*Activate RingIn Tones,type RING & send to  <#> -Cond.apply*',''),
        ]
    )

    if language_choice == 'en':
        print("Using English dataset...")
        with open(SOURCE_DIRECTORY + 'smsCorpus_en_2015.03.09_all.json') as f:
            corpus = json.load(f)
    else:
        print("Using Chinese dataset")
        with open(SOURCE_DIRECTORY + 'smsCorpus_zh_2015.03.09.json') as f:
            corpus = json.load(f)

    texts = []
    for i, text in enumerate(corpus['smsCorpus']['message']):
        try:
            message_raw = str(text['text']['$'])
            cleaned_msg = replace_all(message_raw, od)
            if cleaned_msg not in texts:
                texts.append(cleaned_msg)
            
        except KeyError:
            print("KeyError in row number: ", i)
            continue

    lengths = [len(text) for text in texts]
    sequence_length = mode(lengths)

    print("Number of Texts:", len(corpus['smsCorpus']['message']))
    print("Sequence Length: ", sequence_length)

    filtered_texts = [text for text in texts if len(text) >= sequence_length]
    print("Number of Filtered Texts:", len(filtered_texts))

    vocab = set()
    for text in filtered_texts:
        vocab.update(text)
    vocab_size = len(vocab)+1
    print("Vocab Size:", vocab_size)

    number_of_words = sum(lengths)
    print("Total words", number_of_words)
    print("Vocab / Total words ratio:", round(vocab_size/number_of_words, 3))

    if language_choice == "en":
        with open(os.path.join(SOURCE_DIRECTORY, "nus_en.txt"), 'w') as f:
            for line in filtered_texts[:-1]:
                f.write(line + "\n")
            f.write(filtered_texts[-1])
    else:
        with open(os.path.join(SOURCE_DIRECTORY, "nus_cn.txt"), 'w') as f:
            for line in filtered_texts[:-1]:
                f.write(line + "\n")
            f.write(filtered_texts[-1])

def prep_seame(language_choice='en'):
    """
    Relatively straightforward, just remove the tags
    """

    od = OrderedDict(
        [
            ("<v-noise>", "")
        ]
    )
    
    SOURCE_DIRECTORY = './data/3_seame/'

    corpus = []

    if language_choice == 'en':
        print("Using English dataset...")
        f = open(SOURCE_DIRECTORY + 'seame_sge.txt')
    else:
        print("Using Chinese dataset")
        f = open(SOURCE_DIRECTORY + 'seame_man.txt')

    for line in f.readlines():
        removed_front_id = ' '.join(line.split(' ')[1:]).strip()
        removed_tags = replace_all(removed_front_id, od)
        corpus.append(removed_tags.strip())

    if language_choice == "en":
        with open(os.path.join(SOURCE_DIRECTORY, "seame_en.txt"), 'w') as f:
            for line in corpus[:-1]:
                f.write(line + "\n")
            f.write(corpus[-1])
    else:
        with open(os.path.join(SOURCE_DIRECTORY, "seame_cn.txt"), 'w') as f:
            for line in corpus[:-1]:
                f.write(line + "\n")
            f.write(corpus[-1])

def prep_sbc():
    """
    Dataset from: https://childes.talkbank.org/access/Biling/Singapore.html

    pylangacq http://pylangacq.org/tutorial.html#questions-issues
    nltk http://www.nltk.org/howto/childes.html (not used here)
    https://www.nltk.org/_modules/nltk/corpus/reader/childes.html
    """
    import pylangacq as pla

    od = OrderedDict(
        [
            ("@s", ""),
            ("@o", ""),
            (":zho", ""),
        ]
    )
    SOURCE_DIRECTORY = './data/4_sbc/'

    corpus = []

    for f in os.listdir(SOURCE_DIRECTORY):
        if ".cha" in f:

            raw_cha = pla.read_chat(os.path.join(SOURCE_DIRECTORY,f))
            for utt in raw_cha.utterances():
                if "@si" not in utt[1]: # @si are routine, repeated utterances
                    removed_annotations = replace_all(utt[1], od)
                    if removed_annotations not in corpus:  # remove duplicates ; 
                        corpus.append(removed_annotations)

    with open(os.path.join(SOURCE_DIRECTORY, "sbc.txt"), 'w') as f:
        for line in corpus[:-1]:
            f.write(line + "\n")
        f.write(corpus[-1])

def prep_malaya():
    """
    Dataset from: https://github.com/huseinzol05/Malaya-Dataset

    https://github.com/aosingh/lexpy

    https://github.com/dwyl/english-words
    https://www.kaggle.com/rtatman/english-word-frequency/data
    https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset
    
    https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace
    """
    
    import pickle
    import re
    from lexpy.trie import Trie

    SOURCE_DIRECTORY = './data/5_malaya/'
    
    corpus_cm = [] # keep only lines with code mixing

    with open("./data/english.pickle", "rb") as input_file:
        en_words = pickle.load(input_file)[0]

    print("Trie (english) loaded!")

    with open("./data/singlish.pickle", "rb") as input_file:
        sing_words = pickle.load(input_file)[0]

    print("Trie (singlish) loaded!")

    with open(os.path.join(SOURCE_DIRECTORY, 'singlish.txt')) as f:
        for i, line in enumerate(f.readlines()):

            if 'Players Alive:' in line or 'Share the wisdom:  FB | TW⚖ Rate this question:' in line:
                continue
            # if re.findall(r'[\u4e00-\u9fff]+', line) and re.findall(r'[a-zA-Z]+', line):
            #     corpus_cm.append(line.strip())

            # we want to check for code-mixing in pinyin. once there's a non-english word, we assume it might be what we want
            # on hindsight, could have just matched them against a pinyin database...
            
            cleaned_line = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               line)
            cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line) # remove multiple spaces
            cleaned_line = re.sub('[^a-zA-Z ]+', ' ', cleaned_line) # keep only alphas
            
            possible_singlish = False
            for word in cleaned_line.strip().split(' '):
                if word.lower() not in en_words and word.lower() not in sing_words:
                    if not word.isupper(): # likely to be product model number or acronym
                        possible_singlish = True
                        break
            
            if possible_singlish or (re.findall(r'[\u4e00-\u9fff]+', line) and re.findall(r'[a-zA-Z]+', line)):
                corpus_cm.append(line.strip())

            if i % 1000000 == 0:
                print(i)

    with open(os.path.join(SOURCE_DIRECTORY, "malaya5.txt"), 'w') as f:
        for line in corpus_cm[:-1]:
            f.write(line + "\n")
        f.write(corpus_cm[-1]) 

def prep_combined(only_en=False, cn_default='pinyin'):
    """
    only_en: if True, keeps only English parts. cn_default does not do anything in that case. 

    cn_default: Default translation for Chinese. Either 'pinyin' or 'script'
    - pinyin means romanized transliteration of the Chinese word will be used
    - script means the Simplified/Traditional Chinese character will be retained, whenever available
    """

    import re 
    from pypinyin import pinyin, Style
    from hanziconv import HanziConv # trad chinese to simplified chinese
 
    SOURCE_DIRECTORY = './data/'

    nsc_replacements = [
        ('<malay>[^<>]*<\/malay>', ''), # <malay>[^<>]*<\/malay> # remove malay words
        ('<malay>[^<>]*<malay>', ''),
        ('<malay<[^<>]*<\/malay', ''), # missing > 
        ('\w+~', ''), # \w+~ # remove incomplete words 
        ('\(\w+\)', ''), # \(\w+\) # remove fillers
        ('\[\w+\]', ''), # \[\w+\] # remove discourse particles
        ('!\w+!', ''), # !\w+! # remove interjections
        # ('#[\w| ]+#', ''), # #\w+# # remove foreign words
        ('-\w*#?', ''), # -\w*#? # remove trailing - because of the foreign word removal
        ('\w*-', ''),
        ('#', ''),
    ]

    mandarin_replacements = [
        ('<mandarin>[^<>]*<\/mandarin>', ''), # actual 
        ('<mandairn>[^<>]*<\/mandarin>', ''), # mispelling 
        ('<mandarin>[^<>]*<mandarin>', ''), # missing /
        ('<mandarin<[^<>]*<\/mandarin', ''), # missing > 
    ]

    def mandarin_cleaning(old, new, cleaned_line, only_en, cn_default):
        if only_en:
            cleaned_line = re.sub(old, new, cleaned_line)
        else:
            match = re.search(old, cleaned_line)
            if match:
                chinese_parts = match.group(0) # retrieve chinese parts out, e.g. <mandarin>没有:mei you</mandarin>
                chinese_parts = re.sub('<\/?\w*>','', chinese_parts) # remove the <mandarin> and </mandarin> tag
                chinese_parts_list = chinese_parts.split(':')
                if len(chinese_parts_list) == 1:
                    chinese_parts_list = chinese_parts.split('：')
                try:
                    if cn_default == 'pinyin': # uses a library to convert`没有` to `mei2 you3` 
                        romanized = pinyin(chinese_parts_list[0], style=Style.TONE3)
                        flat_list = [item for sublist in romanized for item in sublist]
                        cleaned_line = re.sub(old, ' '.join(flat_list), cleaned_line)
                    else:
                        cleaned_line = re.sub(old, chinese_parts_list[0], cleaned_line)
                        cleaned_line = HanziConv.toSimplified(cleaned_line)

                except IndexError: # missing delimiter

                    if chinese_parts_list[0].isalpha(): # if only pinyin available, we throw everything away since their pinyin is uninformative
                        cleaned_line = re.sub(old, new, cleaned_line)
                    else: # might have both or chinese only. nevertheless, chinese is always in front
                        chinese_characters = chinese_parts_list[0].split(' ')
                        if cn_default == 'pinyin':
                            cleaned_line = re.sub(old, ' '.join(pinyin(chinese_characters[0], style=Style.TONE3)), cleaned_line)
                        else:
                            cleaned_line = HanziConv.toSimplified(chinese_characters[0])

        return cleaned_line

    def uniform_cleaning(cleaned_line, only_en, cn_default):

        cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line)
        cleaned_line = re.sub(r'[^\u4e00-\u9fff|A-Z|a-z| |,|.]+', '', cleaned_line) # keep only ; goal is to remove num and emoji and funny punct. dw ! ? too 
        if only_en: # replace all cn with empty str
            cleaned_line = re.sub(r'[\u4e00-\u9fff]+','',cleaned_line) 
        else: 
            if cn_default == 'pinyin': # convert to numbered pinyin
                romanized = pinyin(cleaned_line, style=Style.TONE3)
                flat_list = [item for sublist in romanized for item in sublist]
                cleaned_line = ' '.join(flat_list)
            else: # keep chinese script
                cleaned_line = HanziConv.toSimplified(cleaned_line)

        return cleaned_line

    nsc = []
    with open(SOURCE_DIRECTORY + '1_nsc/nsc_conversations.txt') as f:
        for line in f:
            cleaned_line = line

            for old, new in nsc_replacements:
                cleaned_line = re.sub(old, new, cleaned_line)

            for old, new in mandarin_replacements:
                cleaned_line = mandarin_cleaning(old, new, cleaned_line, only_en, cn_default)

            cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line)

            nsc.append(cleaned_line.strip().lower())

    nus_cn = []
    with open(SOURCE_DIRECTORY + '2_nus_sms/nus_cn.txt') as f:
        for line in f:
            cleaned_line = line
            cleaned_line = uniform_cleaning(cleaned_line, only_en, cn_default)
            nus_cn.append(cleaned_line.strip().lower())

    nus_en = []
    with open(SOURCE_DIRECTORY + '2_nus_sms/nus_en.txt') as f:
        for line in f:
            cleaned_line = line
            cleaned_line = uniform_cleaning(cleaned_line, only_en, cn_default)
            nus_en.append(cleaned_line.strip().lower())

    seame_cn = []
    with open(SOURCE_DIRECTORY + '3_seame/seame_cn.txt') as f:
        for line in f:
            cleaned_line = line
            cleaned_line = uniform_cleaning(cleaned_line, only_en, cn_default)
            seame_cn.append(cleaned_line.strip().lower())

    seame_en = []
    with open(SOURCE_DIRECTORY + '3_seame/seame_en.txt') as f:
        for line in f:
            cleaned_line = line
            cleaned_line = uniform_cleaning(cleaned_line, only_en, cn_default)
            seame_en.append(cleaned_line.strip().lower())

    sbc = []
    with open(SOURCE_DIRECTORY + '4_sbc/sbc.txt') as f:
        for line in f:
            cleaned_line = line
            cleaned_line = uniform_cleaning(cleaned_line, only_en, cn_default)
            sbc.append(cleaned_line.strip().lower())

    malaya = []
    with open(SOURCE_DIRECTORY + '5_malaya/malaya.txt') as f:
        for line in f:
            cleaned_line = line
            cleaned_line = uniform_cleaning(cleaned_line, only_en, cn_default)
            malaya.append(cleaned_line.strip().lower())

    combined = nsc + nus_cn + nus_en + seame_cn + seame_en + sbc + malaya

    print(len(nsc))

    print(len(nus_cn))
    print(len(nus_en))

    print(len(seame_cn))
    print(len(seame_en))

    print(len(sbc))

    print(len(malaya))

    print("before removing dup:", len(combined))

    combined = list(set(combined)) # remove duplicates but lose order...

    print("after removing dup:", len(combined))

    if only_en:

        with open(os.path.join(SOURCE_DIRECTORY, "combined_en.txt"), 'w') as f:
            for line in combined[:-1]:
                f.write(line + "\n")
            f.write(combined[-1])

    else:

        if cn_default == 'pinyin':

            with open(os.path.join(SOURCE_DIRECTORY, "combined_pinyin.txt"), 'w') as f:
                for line in combined[:-1]:
                    f.write(line + "\n")
                f.write(combined[-1])

        else:

            with open(os.path.join(SOURCE_DIRECTORY, "combined_script.txt"), 'w') as f:
                for line in combined[:-1]:
                    f.write(line + "\n")
                f.write(combined[-1])

            combined_strict = []
            for cleaned_line in combined:
                # cleaned_line = line.strip()
                if re.findall(r'[\u4e00-\u9fff]+', cleaned_line) and re.findall(r'[a-zA-Z]+', cleaned_line):
                    combined_strict.append(cleaned_line.strip())

            print(len(combined_strict))

            with open(os.path.join(SOURCE_DIRECTORY, "combined_strict.txt"), 'w') as f:
                for line in combined_strict[:-1]:
                    f.write(line + "\n")
                f.write(combined_strict[-1])

def prep_words(choice='singlish'):
    """
    singlish_words.txt from https://github.com/tlkh/singlish-vocab/blob/master/singlish.txt
    singlish.dictionary from https://github.com/3jmaster/singlish-dictionary/blob/master/singlish.dictionary
    phrases.txt from https://github.com/dvrylc/is-singlish/blob/master/phrases.js

    https://en.wikipedia.org/wiki/Singlish_vocabulary
    http://www.mysmu.edu/faculty/jacklee/
    https://www.aussiepete.com/2008/05/singlish-language-guide-for-foreigners.html
    """

    import pickle
    import re
    from lexpy.trie import Trie

    SOURCE_DIRECTORY = './data/' + choice + '/'

    corpus = Trie()

    for txtfile in os.listdir(SOURCE_DIRECTORY):
        if '.txt' in txtfile:
            with open(os.path.join(SOURCE_DIRECTORY, txtfile), 'r') as f:
                for line in f.readlines():
                    cleaned_line = re.sub('[^a-zA-Z ]+', '', line) # keep only alphas
                    cleaned_line_parts = cleaned_line.split(' ')
                    for part in cleaned_line_parts:
                        if part not in corpus:
                            corpus.add(part) 

            print(txtfile, 'done!')

    print(corpus.get_word_count(), 'words in the Trie')

    with open("./data/" + choice + ".pickle", "wb") as output_file:
        pickle.dump([corpus], output_file)

    return corpus

def generate_data(filename, pretrained=False, jieba_tokenisation=False):
    """
    Generate the training, validation, and test datasets
    """

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    import jieba

    import json

    SOURCE_DIRECTORY = './data/'
    TARGET_DIRECTORY = './'
    
    np.random.seed(42)
    
    dataset = []
    with open(SOURCE_DIRECTORY + filename) as f: # combined_strict_script 
        for line in f:
            dataset.append(line.strip())

    labels = []
    with open(SOURCE_DIRECTORY + 'combined_strict/labels.txt') as f: 
        for line in f:
            labels.append(line.strip())

    print("len(labels): ", len(labels))
    num_classes = len(list(set(labels)))
    print("num_classes: ", num_classes)

    X = dataset
    y = labels

    if pretrained:

        metadata = []

        tokenizer = pretrained.tokenizer # bert_embedding.tokenizer
        sentences_tokenized = []
        for sentence in X:
            sentence_tokenized = tokenizer.tokenize(sentence)
            sentences_tokenized.append(sentence_tokenized)

        X = sentences_tokenized

    else:
    
        if 'script' in filename and jieba_tokenisation: # chinese char only

            X = []
            for doc in dataset:
                seg_list = jieba.cut(doc, cut_all=False)
                seg_list = list(seg_list)
                X.append(seg_list)

            import scipy.stats as stats
            import pylab as pl
            h = sorted([len(sentence) for sentence in X])
            maxLength = h[int(len(h) * 0.95)]
            print("Max length is: ",h[len(h)-1])
            print("95% cover length up to: ", maxLength)

            X = [" ".join(wordslist) for wordslist in X]  # Keras Tokenizer expect the words tokens to be seperated by space 
            input_tokenizer = Tokenizer() # Initial vocab size
            input_tokenizer.fit_on_texts(X)
            vocab_size = len(input_tokenizer.word_index) + 1
            print("input vocab_size:",vocab_size)
            X = np.array(pad_sequences(input_tokenizer.texts_to_sequences(X), maxlen=maxLength))
            
            tokenizer_json = input_tokenizer.to_json() 
            with open(TARGET_DIRECTORY + 'senti_base_cn_tokenizer.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

            target_tokenizer = Tokenizer() # 3?
            target_tokenizer.fit_on_texts(y)
            print("output vocab_size:",len(target_tokenizer.word_index) + 1)
            y = np.array(target_tokenizer.texts_to_sequences(y)) -1
            y = y.reshape(y.shape[0])
            y = to_categorical(y, num_classes=num_classes)

            target_reverse_word_index = {v: k for k, v in list(target_tokenizer.word_index.items())}
            sentiment_tag = [target_reverse_word_index[1],target_reverse_word_index[2]] 
            metadata = {
                "maxLength": maxLength,
                "vocab_size": vocab_size,
                "output_dimen": num_classes,
                "sentiment_tag": sentiment_tag
            }

        elif 'pinyin' in filename: # en with pinyin

            metadata = []

        else: # both

            metadata = []

    # 60, 20, 20 split 
    from sklearn.model_selection import train_test_split 
    X_nontrain, X_test, y_nontrain, y_test = train_test_split(X, y, stratify=labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_nontrain, y_nontrain, stratify=y_nontrain, test_size=0.25, random_state=42)

    return (X_train, X_val, X_test, y_train, y_val, y_test, metadata)

def convert_to_pinyin(filename):
    """
    Take in a dataset, read line by line, use pinyin library to convert 
    Chinese script to numbered pinyin, save the file with _pinyin appended
    """
    
    from pypinyin import pinyin, Style

    SOURCE_DIRECTORY = './data/'

    corpus = []
    with open(SOURCE_DIRECTORY + filename) as f:
        for line in f:
            romanized = pinyin(line, style=Style.TONE3)
            flat_list = [item for sublist in romanized for item in sublist]
            cleaned_line = ' '.join(flat_list)
            cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line) # remove multiple spaces
            corpus.append(cleaned_line.strip())

    with open(os.path.join(SOURCE_DIRECTORY, filename[:-4] + "_pinyin.txt"), 'w') as f:
        for line in corpus[:-1]:
            f.write(line + "\n")
        f.write(corpus[-1])

    return corpus

def keep_only_script(filename):
    """
    Keeps only Chinese characters in the text
    """

    SOURCE_DIRECTORY = './data/'

    corpus = []
    with open(SOURCE_DIRECTORY + filename) as f:
        for line in f:
            cleaned_line = re.sub(r'[^\u4e00-\u9fff]+', '', line) # keep only chinese
            cleaned_line = re.sub(r'\ \ +', ' ', cleaned_line) # remove multiple spaces
            corpus.append(cleaned_line.strip())

    with open(os.path.join(SOURCE_DIRECTORY, filename[:-4] + "_script.txt"), 'w') as f:
        for line in corpus[:-1]:
            f.write(line + "\n")
        f.write(corpus[-1])

    return corpus


if __name__ == "__main__":

    # prep_words() # singlish
    # prep_words('english')

    # prep_nsc()
    # prep_nussms('en') # en cn
    # prep_nussms('cn') # en cn
    # prep_seame('en') # en cn
    # prep_seame('cn') # en cn
    # prep_sbc()
    # prep_malaya()
    # prep_combined(cn_default='script') # only_en=False, cn_default='pinyin'
    # prep_combined(cn_default='pinyin') # only_en=False, cn_default='pinyin'
    # prep_combined(only_en=True) # only_en=False, cn_default='pinyin'

    # convert_to_pinyin('combined_strict.txt')
    # convert_to_pinyin('combined_script.txt')
    # keep_only_script('combined_strict.txt')

    generate_data('combined_strict_script.txt')
