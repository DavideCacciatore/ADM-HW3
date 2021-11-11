import ctypes
import glob
import json
import math
import multiprocessing as mp
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def stemSentence(sentence, porter):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return ' '.join(stem_sentence)


def info_pre_processing_f(text):
    # removing punctuation, lowercase the text, removing stopwords
    # text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    p_text = text.translate(translator).lower()
    ppt = ""
    for word in p_text.split():
        if word not in stopwords.words('english'):
            ppt += word + " "
    return ppt.strip(" ")


def to_vocabulary(sentence, vocabulary, inverted_index, complex_index, file_num, v, lock):
    len_sentence = len(sentence)
    for word in sentence.split(" "):
        if len_sentence == 0:
            tf = 0
        else:
            tf = sentence.count(word) / len_sentence
        # w = zlib.adler32(word.encode('utf-8'))
        # if it's the frst time word is processed, it cannot be in inverted index
        if word not in vocabulary:
            with lock:
                v.value += 1
            w = v.value
            vocabulary[word] = w
            inverted_index[w] = ["document_" + file_num]
            complex_index[w] = [("document_" + file_num, tf)]
        else:
            w = vocabulary[word]
            inverted_index[w] = inverted_index[w] + ["document_" + file_num]
            complex_index[w] = complex_index[w] + [("document_" + file_num, tf)]


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


def write_complex_index(complex_index):
    file_path = os.path.join(os.getcwd(), "complex_index.json")
    json_object = json.dumps(complex_index.copy(), indent=4)
    with open(file_path, "w+") as outfile:
        outfile.write(json_object)
    print("Complex index written")


def process_anime(starting_page, path, vocabulary, inverted_index, complex_index, porter, v, lock):
    task_dim = 8
    # last task processes less pages
    if starting_page == 376:
        task_dim = 7
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
            to_vocabulary(p_syn, vocabulary, inverted_index, complex_index, str(index), v, lock)
            print(file + " inserted in vocab")
            index += 1


def parallelize_parsing(path, vocabulary, inverted_index, complex_index, porter, v, lock):
    pool = mp.Pool(mp.cpu_count())
    # 48 tasks : 8 pages per task, last excluded (7)
    page = 0
    for r in range(0, 383, 8):
        pool.apply(process_anime, args=(page + r, path, vocabulary, inverted_index, complex_index, porter, v, lock))
    pool.close()
    pool.join()


def conjunctive_query(path, porter):
    query = input("Search keys : ").split()
    # found docs with each word, then intersect
    all_docs = list()
    j_codes = json.loads(open('vocabulary.json').read())
    j_index = json.loads(open('inverted_index.json').read())
    try:
        for word in query:
            word = porter.stem(word)
            code = j_codes[word]
            print(word, code)
            all_docs.append(set(j_index[str(code)]))
    except KeyError:
        print("No results found")
        exit()
    docs = set.intersection(*all_docs)
    # get a list of doc indeces
    docs = [int(el.split("_")[1]) for el in docs]
    with open("url_of_interest.txt", encoding="utf-8") as u:
        urls = u.readlines()
    table = list()
    for i in docs:
        sub_file = f"data/page_{str(int(i / 50))}/animes_{str(int(i / 50))}/anime_{i}.tsv"
        file = os.path.join(path, sub_file)
        with open(file, encoding="utf-8") as fp:
            t = fp.readlines()
        title = t[1].split("\t")[0]
        synopsis = t[1].split("\t")[10]
        url = urls[i]
        table.append([title, synopsis, url])
    print(tabulate(table, headers=['animeTitle', 'animeDescription', 'Url'], tablefmt="fancy_grid"))
    return docs


def compute_tfIdf(words, start, end, vocab, complex_index_json, complex_index):
    for word in words[start:end]:
        print(word)
        tfIdf_docs = list()
        word_code = vocab[word]
        docs_tfIdf = dict(complex_index_json[str(word_code)])
        # I got the term and the docs associated to it
        # for each doc, compute the tfIdf
        for name in docs_tfIdf.keys():
            i = int(name.split("_")[1])
            tf = docs_tfIdf[name]
            idf = math.log10(19131/len(docs_tfIdf))
            couple = (f"document_{i}", round(tf * idf, 5))
            tfIdf_docs.append(couple)
        complex_index[word_code] = tfIdf_docs


def get_complex_index(complex_index):
    pool = mp.Pool(mp.cpu_count())
    vocab = json.loads(open('vocabulary.json').read())
    complex_index_json = json.loads(open('complex_index.json').read())
    words = list(vocab.keys())
    # 20 tasks : last task process 1 more element
    for pos in range(0, len(words), 1954):
        if pos == 37126:
            end = pos + 1955
        start, end = pos, pos + 1954
        pool.apply(compute_tfIdf, args=[words, start, end, vocab, complex_index_json, complex_index])
    pool.close()
    pool.join()


if __name__ == '__main__':
    path = os.getcwd()
    manager = mp.Manager()
    vocabulary = manager.dict()
    inverted_index = manager.dict()
    complex_index = manager.dict()
    v = manager.Value(ctypes.c_ulonglong, 0)
    lock = manager.Lock()
    # nltk.download('punkt')
    porter = PorterStemmer()
    #parallelize_parsing(path, vocabulary, inverted_index, complex_index, porter, v, lock)
    #write_vocabulary(vocabulary)
    #write_inverted_index(inverted_index)
    #write_complex_index(complex_index)
    #conjunctive_query(path, porter)
    # at this point you have an incomplete complex_index since the number associated to each doc is only the tf part
    get_complex_index(complex_index)
    write_complex_index(complex_index)