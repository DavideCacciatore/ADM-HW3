import glob
import heapq
import json
import math
import multiprocessing as mp
import os
import re
import string

import pandas
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from IPython.display import display
from tabulate import tabulate


def stemSentence(sentence, porter):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return ' '.join(stem_sentence)


def info_pre_processing_f(text):
    # removing punctuation, lowercase the text, removing stopwords
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
        # if it's the first time word is processed, it cannot be in inverted index
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


def conjunctive_query(porter):
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
        synopsis = re.sub("(.{60})", "\\1\n", synopsis, 0, re.DOTALL)
        url = urls[int(d)]
        if type == "tfIdf":
            table.append([title, synopsis, url, similarity])
        else:
            table.append([title, synopsis, url])
    df = pandas.DataFrame(table, columns=headers)
    display(df)
    #print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


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
    heapq.heapify(cos_sim_doc)
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
    for doc in docs:
        numerator, denumerator = 0, 0
        doc_id = doc.split("_")[1]
        synopsis = j_docs_short[doc_id][2]
        for code in query_codes:
            code = str(code)
            word_related_doc = dict(j_complex_index[code])
            tfIdf_q = word_related_doc[doc]
            numerator += tfIdf_q
        debug = list()
        for word in synopsis.split(" "):
            s_code = str(j_codes[word])
            debug.append(s_code)
            word_list = dict(j_complex_index[s_code])
            tfIdf_d = word_list[doc]
            denumerator += tfIdf_d ** 2
        # for each doc compute cos similarity
        cos_sim = numerator / (len(query_codes) * math.sqrt(denumerator))
        heapq.heappush(cos_sim_doc, (round(cos_sim, 5), doc))
    return cos_sim_doc


def cosine_similarity_rank(porter, K):
    query_string = input("Search keys - conjunctive query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    cos_sim_doc = all_tfIdf(query)
    print(cos_sim_doc)
    result = list()
    if len(cos_sim_doc) > K:
        for k in range(K):
            result.append(heapq.heappop(cos_sim_doc))
    else:
        for k in range(len(cos_sim_doc)):
            result.append(heapq.heappop(cos_sim_doc))
    return reversed([(el[1], el[0]) for el in result])


def custom_rank(porter, K):
    query_string = input("Search keys - conjunctive query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    cos_sim_doc = all_tfIdf(query)
    print(cos_sim_doc)
    j_docs_short = json.loads(open('docs_short.json').read())
    values = j_docs_short.values()
    max_members = max([int(v[0]) for v in values])
    max_popularity = max([int(v[1]) for v in values])
    h_rank = list()
    heapq.heapify(h_rank)
    for sim, doc in cos_sim_doc:
        doc_num = doc.split("_")[1]
        mem, pop = int(j_docs_short[doc_num][0]), int(j_docs_short[doc_num][1])
        measure = sim * 0.5 + (1 - pop/max_popularity) * 0.25 + (mem/max_members) * 0.25
        heapq.heappush(h_rank, (measure, doc))
    result = list()
    if len(h_rank) > K:
        for k in range(K):
            q = heapq.heappop(h_rank)
            result.append((q[1], round(q[0], 5)))
    else:
        for k in range(len(h_rank)):
            q = heapq.heappop(h_rank)
            result.append((q[1], round(q[0], 5)))
    return reversed(result)
