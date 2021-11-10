import ctypes
import glob
import json
import multiprocessing as mp
import os
import re
import string
import zlib
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer


def stemSentence(sentence, porter):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return ' '.join(stem_sentence)


def info_pre_processing_f(text):
    # removing punctuation, lowercase the text, removing stopwords
    #text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    p_text = text.translate(translator).lower()
    ppt = ""
    for word in p_text.split():
        if word not in stopwords.words('english'):
            ppt += word + " "
    return ppt.strip(" ")


def to_vocabulary(sentence, vocabulary, inverted_index, file_num, v, lock):
    for word in sentence.split(" "):
        # w = zlib.adler32(word.encode('utf-8'))
        # if it's the frst time word is processed, it cannot be in inverted index
        if word not in vocabulary:
            with lock:
                v.value += 1
            w = v.value
            vocabulary[word] = w
            inverted_index[w] = ["document_" + file_num]
        else:
            w = vocabulary[word]
            inverted_index[w] = inverted_index[w] + ["document_" + file_num]


def write_vocabulary(vocabulary):
    file_path = os.path.join(os.getcwd(), "vocabulary.json")
    json_object = json.dumps(vocabulary.copy(), indent=4)
    with open(file_path, "w+") as outfile:
        outfile.write(json_object)
    print("Vocabulary written")


def write_inverted_index(inverted_index):
    file_path = os.path.join(os.getcwd(), "inverted_index.json")
    json_object = json.dumps(inverted_index.copy(), indent=4)
    with open(file_path, "w+") as outfile:
        outfile.write(json_object)
    print("Inverted index written")


def process_anime(starting_page, path, vocabulary, inverted_index, porter, v, lock):
    task_dim = 28
    # last task processes less pages
    if starting_page == 364:
        task_dim = 19
    for i in range(starting_page, starting_page + task_dim):
        index = i * 50
        # here you put the name of the dir where you have all the folders
        home = os.path.join(path, f"data/page_{i}/animes_{i}")
        for file in glob.iglob(r'' + re.escape(home) + '\*.tsv'):
            with open(file, encoding="utf-8") as fp:
                t = fp.readlines()
            synopsis = t[1].split("\t")[10]
            pp_syn = info_pre_processing_f(synopsis)
            p_syn = stemSentence(pp_syn, porter)
            to_vocabulary(p_syn, vocabulary, inverted_index, str(index), v, lock)
            print(file + " inserted in vocab")
            index += 1


def parallelize_parsing(path, vocabulary, inverted_index, porter, v, lock):
    pool = mp.Pool(mp.cpu_count())
    # 28 tasks : 4 pages per task
    page = 0
    for r in range(0, 383, 28):
        pool.apply(process_anime, args=(page + r, path, vocabulary, inverted_index, porter, v, lock))
    pool.close()
    pool.join()


if __name__ == '__main__':
    manager = mp.Manager()
    vocabulary = manager.dict()
    inverted_index = manager.dict()
    v = manager.Value(ctypes.c_ulonglong, 0)
    lock = manager.Lock()
    # nltk.download('punkt')
    porter = PorterStemmer()
    parallelize_parsing(os.getcwd(), vocabulary, inverted_index, porter, v, lock)
    write_vocabulary(vocabulary)
    write_inverted_index(inverted_index)

    '''
    
    file = "C:/Users/USER/Documents/ADMhmw/data/page_0/animes_0/anime_0.tsv"
    with open(file, encoding="utf-8") as fp:
        t = fp.readlines()
    # print([el for el in zip(t[0].strip("\n").split("\t"), t[1].strip("\n").split("\t"))])
    synopsis = t[1].split("\t")[10]
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    text = synopsis.translate(translator).lower()
    text = ' '.join(text.split())
    #print(synopsis)
    pp_syn = info_pre_processing_f(text)
    p_syn = stemSentence(pp_syn, porter)
    for i in p_syn.split(" "):
        print(i)
    #to_vocabulary(p_syn)'''