import heapq
import json
import math
import os
import re
import pandas
from IPython.display import display

import vocabularize as voc


# pre-process the query to get it the same kind of entity as the synopsis
def clean_query(porter, query):
    pp_query = voc.info_pre_processing_f(query)
    query = voc.stemSentence(pp_query, porter).split(" ")
    return query


# execute a simple query and get results as list
def conjunctive_query(porter):
    # enter the query
    query_string = input("Search keys - conjunctive query : ")
    # pre-process the query
    query = clean_query(porter, query_string)
    all_docs = list()
    # load into Python dicts the necessary information from JSON files
    j_codes = json.loads(open('vocabulary.json').read())
    j_index = json.loads(open('inverted_index.json').read())
    # for each word, get the docs in which it is present and intersect them to get the ones in which all of them are
    for word in query:
        code = j_codes[word]
        documents = set(j_index[str(code)])
        all_docs.append(documents)
    docs = set.intersection(*all_docs)
    # get a list of doc indeces
    docs = [int(el.split("_")[1]) for el in docs]
    return docs


# compute the cosine similarity
def q_cosine_similarity(query):
    # save documents' id in min a heap data structure
    # implement min heap multiplying values by -1
    cos_sim_doc = list()
    heapq.heapify(cos_sim_doc)
    # found docs with each word, then intersect
    all_docs = list()
    query_codes = list()
    # load JSON files to get access to data stored
    j_codes = json.loads(open('vocabulary.json').read())
    j_inv_index = json.loads(open('inverted_index.json').read())
    j_complex_index = json.loads(open('tfIdf_complex_index.json').read())
    j_docs_short = json.loads(open('docs_short.json').read())
    # get the docs in which each word in query is present
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
        heapq.heappush(cos_sim_doc, (round(cos_sim, 5)*(-1), doc))
    return cos_sim_doc


def cosine_similarity_rank(porter, K):
    query_string = input("Search keys - conjunctive query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    cos_sim_doc = q_cosine_similarity(query)
    print(cos_sim_doc)
    result = list()
    if len(cos_sim_doc) > K:
        for k in range(K):
            result.append(heapq.heappop(cos_sim_doc))
    else:
        for k in range(len(cos_sim_doc)):
            result.append(heapq.heappop(cos_sim_doc))
    return [(el[1], el[0]*(-1)) for el in result]


def custom_rank(porter, K):
    query_string = input("Search keys - custom query : ")
    # found docs with each word, then intersect
    query = clean_query(porter, query_string)
    cos_sim_doc = q_cosine_similarity(query)
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
        heapq.heappush(h_rank, (measure*(-1), doc))
    result = list()
    if len(h_rank) > K:
        for k in range(K):
            q = heapq.heappop(h_rank)
            result.append((q[1], round(q[0], 5)))
    else:
        for k in range(len(h_rank)):
            q = heapq.heappop(h_rank)
            result.append((q[1], round(q[0], 5)))
    return [(el[0], el[1]*(-1)) for el in result]


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