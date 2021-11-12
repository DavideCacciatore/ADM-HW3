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
    list_sentence = sentence.split(" ")
    len_sentence = len(list_sentence)
    for word in list_sentence:
        if len_sentence == 0:
            tf = 0
        else:
            tf = list_sentence.count(word) / len_sentence
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


def write_index(index, index_name):
    file_path = os.path.join(os.getcwd(), f"{index_name}.json")
    json_object = json.dumps(index.copy(), indent=4)
    with open(file_path, "w+") as outfile:
        outfile.write(json_object)
    print("Index written")


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


def parallelize_process_anime(path, vocabulary, inverted_index, complex_index, porter, v, lock):
    pool = mp.Pool(mp.cpu_count())
    # 48 tasks : 8 pages per task, last excluded (7)
    page = 0
    for r in range(0, 383, 8):
        pool.apply(process_anime, args=(page + r, path, vocabulary, inverted_index, complex_index, porter, v, lock))
    pool.close()
    pool.join()


def clean_query(porter, query):
    pp_query = info_pre_processing_f(query)
    query = stemSentence(pp_query, porter).split(" ")
    return query


def conjunctive_query(path, porter):
    query_string = input("Search keys - conjunctive query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    all_docs = list()
    j_codes = json.loads(open('vocabulary.json').read())
    j_index = json.loads(open('inverted_index.json').read())
    for word in query:
        code = j_codes[word]
        documents = set(j_index[str(code)])
        all_docs.append(documents)
    docs = set.intersection(*all_docs)
    # get a list of doc indeces
    docs = [int(el.split("_")[1]) for el in docs]
    return docs


def print_tables(docs, path, type):
    with open("url_of_interest.txt", encoding="utf-8") as u:
        urls = u.readlines()
    table = list()
    for i in docs:
        if type == "tfIdf":
            d, similarity = i[0].split("_")[1], i[1]
            headers = ['animeTitle', 'animeDescription', 'Url', "Similarity"]
        else:
            d = i
            headers = ['animeTitle', 'animeDescription', 'Url']
        p = str(int(int(d) / 50))
        sub_file = f"data/page_{p}/animes_{p}/anime_{d}.tsv"
        file = os.path.join(path, sub_file)
        with open(file, encoding="utf-8") as fp:
            t = fp.readlines()
        title = t[1].split("\t")[0]
        synopsis = t[1].split("\t")[10]
        url = urls[int(d)]
        if type == "tfIdf":
            table.append([title, synopsis, url, similarity])
        else:
            table.append([title, synopsis, url])
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def compute_tfIdf(words, start, end, vocab, tf_j_index, complex_index):
    for word in words[start:end]:
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


def get_complex_index(complex_index):
    pool = mp.Pool(mp.cpu_count())
    vocab = json.loads(open('vocabulary.json').read())
    tf_j_index = json.loads(open('tf_complex_index.json').read())
    words = list(vocab.keys())
    # 20 tasks : last task process 1 more element
    for pos in range(0, len(words), 1954):
        if pos == 37126:
            end = pos + 1955
        start, end = pos, pos + 1954
        pool.apply(compute_tfIdf, args=[words, start, end, vocab, tf_j_index, complex_index])
    pool.close()
    pool.join()


def write_docs_short(docs_short):
    file_path = os.path.join(os.getcwd(), "docs_short.json")
    json_object = json.dumps(docs_short.copy(), indent=4)
    with open(file_path, "w+") as outfile:
        outfile.write(json_object)
    print("Docs short written")


def get_docs_short(starting_page, path, docs_short, porter):
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
            members = t[1].split("\t")[5]
            popularity = t[1].split("\t")[9]
            synopsis = t[1].split("\t")[10]
            pp_syn = info_pre_processing_f(synopsis)
            p_syn = stemSentence(pp_syn, porter)
            docs_short[index] = (members, popularity, p_syn)
            print(f"written in dict {index}")
            index += 1


def parallelize_docs_short(path, docs_short, porter):
    pool = mp.Pool(mp.cpu_count())
    # 48 tasks : 8 pages per task, last excluded (7)
    page = 0
    for r in range(0, 383, 8):
        pool.apply(get_docs_short, args=(page + r, path, docs_short, porter))
    pool.close()
    pool.join()


def all_tfIdf(query):
    cos_sim_doc = list()
    # found docs with each word, then intersect
    all_docs = list()
    query_codes = list()
    j_codes = json.loads(open('vocabulary.json').read())
    j_inv_index = json.loads(open('inverted_index.json').read())
    j_complex_index = json.loads(open('tfIdf_complex_index.json').read())
    j_docs_short = json.loads(open('docs_short.json').read())
    for word in query:
        code = j_codes[word]
        query_codes.append(code)
        all_docs.append(set(j_inv_index[str(code)]))
    docs = set.intersection(*all_docs)
    # numerator --> sum of the tfIdf of the words in the query & in the document
    # since we got the docs from the search we know that the query words are in the docs
    print(docs)
    for doc in docs:
        tfIdf_query_doc = list()
        tfIdf_doc = list()
        doc_number = doc.split("_")[1]
        synopsis = j_docs_short[doc_number][2]
        for code in query_codes:
            code = str(code)
            word_list = dict(j_complex_index[code])
            tfIdf_q = word_list[doc]
            tfIdf_query_doc.append(tfIdf_q)
        for word in synopsis.split(" "):
            s_code = str(j_codes[word])
            word_list = dict(j_complex_index[s_code])
            tfIdf_d = word_list[doc]
            tfIdf_doc.append(tfIdf_d)
        # for each doc compute cos similarity
        tfIdf_squares = [el ** 2 for el in tfIdf_doc]
        print(len(set(query)), len(query))
        print(tfIdf_query_doc)
        print(sum(tfIdf_query_doc))
        print((len(query_codes) * math.sqrt(sum(tfIdf_squares))))
        cos_sim = sum(tfIdf_query_doc) / (len(query_codes) * math.sqrt(sum(tfIdf_squares)))
        cos_sim_doc.append((doc, round(cos_sim, 5)))
    return cos_sim_doc


def cosine_similarity_rank(porter, K):
    query_string = input("Search keys - conjunctive query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    cos_sim_doc = all_tfIdf(query)
    cos_sim_doc.sort(reverse=True, key=lambda x: x[1])
    if len(cos_sim_doc) > K:
        return cos_sim_doc[0:K]
    else:
        return cos_sim_doc


def custom_rank(porter, K):
    query_string = input("Search keys - conjunctive query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    cos_sim_doc = all_tfIdf(query)
    cos_sim_doc.sort(reverse=True, key=lambda x: x[1])
    if len(cos_sim_doc) > K:
        return cos_sim_doc[0:K]
    else:
        return cos_sim_doc


if __name__ == '__main__':
    path = os.getcwd()
    manager = mp.Manager()
    vocabulary = manager.dict()
    inverted_index = manager.dict()
    complex_index = manager.dict()
    docs_short = manager.dict()
    v = manager.Value(ctypes.c_ulonglong, 0)
    lock = manager.Lock()
    # nltk.download('punkt')
    porter = PorterStemmer()
    '''parallelize_process_anime(path, vocabulary, inverted_index, complex_index, porter, v, lock)
    write_index(vocabulary, "vocabulary")
    write_index(inverted_index, "inverted_index")
    write_index(complex_index, "tf_complex_index")
    #################################################################################################################
    # at this point you have an incomplete complex_index since the number associated to each doc is only the tf part
    get_complex_index(complex_index)
    write_index(complex_index, "tfIdf_complex_index")'''
    # parallelize_docs_short(path, docs_short, porter)
    # write_docs_short(docs_short)

    #docs = conjunctive_query(path, porter)
    #print_tables(docs, path, "")
    top = cosine_similarity_rank(porter, 5)
    print_tables(top, path, "tfIdf")
