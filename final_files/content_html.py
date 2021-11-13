import time
import multiprocessing as mp
import bs4
import requests
import os


def get_html(urls, page_id, base_wd, start, end):
    session = requests.Session()
    # Creating a new folder
    directory = f"page_{page_id}"
    # Parent Directories
    parent_dir = base_wd
    # Path
    path = os.path.join(parent_dir, directory)
    # Create the directory
    os.makedirs(path)
    os.chdir(path)
    #print("Directory '%s' created" % directory)
    if end > len(urls):
        end = len(urls)
    for i in range(start, end):
        print(start, end, i)
        URL = urls[i]
        #print[URL]
        page = session.get(URL)
        # Parsing the page
        soup_data = bs4.BeautifulSoup(page.content, "html.parser")
        with open(f"article_{i}.html", "w", encoding="utf-8") as file:
            file.write(str(soup_data))
        #print(f"Article {i} successfully written!")
        time.sleep(2)


def get_content():
    pool = mp.Pool(mp.cpu_count())
    base_wd = os.getcwd()
    with open('url_of_interest.txt', encoding="utf-8") as f:
        urls = [el.strip("\n") for el in f.readlines()]
    # from 13500 till the end
    # 13 rounds per 4 cores ---> 51 tasks (last time 1 core rest)
    page_id = 270
    for task_num in range(0, 5631, 50):
        start, end = page_id * 50, (page_id + 1) * 50
        print(page_id)
        args = (urls, page_id, base_wd, start, end)
        pool.apply(get_html, args)
        page_id += 1
    pool.close()