import glob
import json
import math
import multiprocessing as mp
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# get the stemmed sentence : tokenize and find the stem for each word in sentence
def stemSentence(sentence, porter):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return ' '.join(stem_sentence)


# pre-process a text : return the text as a string
def info_pre_processing_f(text):
    # removing punctuation, lowercase the text, removing stopwords
    # map punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    p_text = text.translate(translator).lower()
    ppt = ""
    for word in p_text.split():
        if word not in stopwords.words('english'):
            ppt += word + " "
    return ppt.strip(" ")


# add to the shared vocbulary the data collected
def to_vocabulary(sentence, vocabulary, inverted_index, complex_index, file_num, v, lock):
    # get the length of the sentence to compute tf value
    list_sentence = sentence.split(" ")
    len_sentence = len(list_sentence)
    # for each word in sentence, compute the tf, then memorize it in the shared dicts
    for word in list_sentence:
        if len_sentence == 0:
            tf = 0
        else:
            tf = list_sentence.count(word) / len_sentence
        # if it's the first time word is processed, it cannot be in the shared dicts
        if word not in vocabulary:
            # get a shared counter value as word identifier
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


# write data in a shared Python dict to a JSON file to store it
def write_index(index, index_name):
    file_path = os.path.join(os.getcwd(), f"{index_name}.json")
    json_object = json.dumps(index.copy(), indent=4)
    with open(file_path, "w+") as outfile:
        outfile.write(json_object)
    print("Index written")


# task function to process each file and populate the shared dicts
def process_anime(starting_page, path, vocabulary, inverted_index, complex_index, porter, v, lock):
    # each task works with 8 pages
    task_dim = 8
    # last task processes less pages (7)
    if starting_page == 376:
        task_dim = 7
    for i in range(starting_page, starting_page + task_dim):
        index = i * 50
        # here you put the name of the dir where you have all the folders
        home = os.path.join(path, f"data/page_{i}/animes_{i}")
        # for each tsv fileextract the information needed to create the indeces files
        for file in glob.iglob(r'' + re.escape(home) + '\*.tsv'):
            with open(file, encoding="utf-8") as fp:
                t = fp.readlines()
            synopsis = t[1].split("\t")[10]
            pp_syn = info_pre_processing_f(synopsis)
            p_syn = stemSentence(pp_syn, porter)
            # call function to populate shared dicts
            to_vocabulary(p_syn, vocabulary, inverted_index, complex_index, str(index), v, lock)
            print(file + " inserted in vocab")
            index += 1


# function to handle parallelization
def parallelize_process_anime(path, vocabulary, inverted_index, complex_index, porter, v, lock):
    # create a pool to be later populated and define a coherent number of tasks to exploit parallelization better
    pool = mp.Pool(mp.cpu_count())
    # 48 tasks : 47 * 8 + 7 : 8 pages per task, last excluded (7)
    page = 0
    # start a task every 28 pages
    for r in range(0, 383, 8):
        pool.apply(process_anime, args=(page + r, path, vocabulary, inverted_index, complex_index, porter, v, lock))
    # close the pool and join the shared results
    pool.close()
    pool.join()


# task function
def compute_tfIdf(words, start, end, vocab, tf_j_index, complex_index):
    # for each word in the parallel slice, compute the tfIdf
    for word in words[start:end]:
        # for each document in which the word is present
        tfIdf_docs = list()
        word_code = vocab[word]
        doc_tf = tf_j_index[str(word_code)]
        docs_tfIdf = dict(doc_tf)
        # I got the term and the docs associated to it : for each doc, compute the tfIdf
        for name in docs_tfIdf.keys():
            i = int(name.split("_")[1])
            tf = docs_tfIdf[name]
            idf = math.log10(19131 / len(doc_tf))
            couple = (f"document_{i}", round(tf * idf, 5))
            tfIdf_docs.append(couple)
        complex_index[word_code] = tfIdf_docs


# function to handle parallelization : get the complete tfIdf complex index
def get_complex_index(complex_index):
    # create a pool to be later populated and define a coherent number of tasks to exploit parallelization better
    pool = mp.Pool(mp.cpu_count())
    # load json files to Python dicts to het access to data already computed stored (words codes, tf)
    vocab = json.loads(open('vocabulary.json').read())
    tf_j_index = json.loads(open('tf_complex_index.json').read())
    words = list(vocab.keys())
    # for each word, compute the tfIdf related to a particular document
    # 20 tasks : last task process 1 more element
    for pos in range(0, len(words), 1954):
        if pos == 37126:
            end = pos + 1955
        start, end = pos, pos + 1954
        # start task to compute the tfIdf
        pool.apply(compute_tfIdf, args=[words, start, end, vocab, tf_j_index, complex_index])
    # close the pool and join the shared results
    pool.close()
    pool.join()


# task function : get the document index
def get_docs_short(starting_page, path, docs_short, porter):
    task_dim = 8
    # last task processes less pages
    if starting_page == 376:
        task_dim = 7
    for i in range(starting_page, starting_page + task_dim):
        index = i * 50
        # here you put the name of the dir where you have all the folders
        home = os.path.join(path, f"data/page_{i}/animes_{i}")
        # for each tsv in the directory, extract the needed information
        for file in glob.iglob(r'' + re.escape(home) + '\*.tsv'):
            with open(file, encoding="utf-8") as fp:
                t = fp.readlines()
            members = t[1].split("\t")[5]
            popularity = t[1].split("\t")[9]
            synopsis = t[1].split("\t")[10]
            pp_syn = info_pre_processing_f(synopsis)
            p_syn = stemSentence(pp_syn, porter)
            # store the data in the shared dict
            docs_short[index] = (members, popularity, p_syn)
            print(f"written in dict {index}")
            index += 1


# function to handle parallelization : get the index whose keys are documents' identifiers
# this index will be used to compute the customized ranking
def parallelize_docs_short(path, docs_short, porter):
    # create a pool to be later populated and define a coherent number of tasks to exploit parallelization better
    pool = mp.Pool(mp.cpu_count())
    # 48 tasks : 47 * 8 + 7 : 8 pages per task, last excluded (7)
    page = 0
    # start a task every 28 pages
    for r in range(0, 383, 8):
        pool.apply(get_docs_short, args=(page + r, path, docs_short, porter))
    # close the pool and join the shared results
    pool.close()
    pool.join()

