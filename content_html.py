import time
import multiprocessing as mp
import bs4
import requests
import os


# create a directory for each page ( 50 links ) and store there the corresponding html files
def get_html(urls, page_id, base_wd, start, end):
    session = requests.Session()
    # Creating a new folder
    directory = f"page_{page_id}"
    # Parent Directories
    parent_dir = base_wd
    # Path
    path = os.path.join(parent_dir, directory)
    # Create the directory and change the cwd there to create and store files there
    os.makedirs(path)
    os.chdir(path)
    # last task works on 31 files instead of 50: check it to avoid bounding errors
    if end > len(urls):
        end = len(urls)
    for i in range(start, end):
        # get the i-th url and retrieve the html file
        URL = urls[i]
        page = session.get(URL)
        # Parsing the page
        soup_data = bs4.BeautifulSoup(page.content, "html.parser")
        # create and store it in the directory
        with open(f"article_{i}.html", "w", encoding="utf-8") as file:
            file.write(str(soup_data))
        print(f"Article {i} successfully written!")
        # sleep 2 seconds to let other tasks work too and avoid connection errors
        time.sleep(2)


# function to handle the parallelization
def get_content():
    # create a pool to be later populated and define a coherent number of tasks to exploit parallelization better
    # get current directory
    pool = mp.Pool(mp.cpu_count())
    base_wd = os.getcwd()
    with open('url_of_interest.txt', encoding="utf-8") as f:
        urls = [el.strip("\n") for el in f.readlines()]
    # start a task for each page (50 links)
    # 96 rounds per 4 cores ---> 382 tasks (last task will exploit only 2 cores)
    for page_id in range(383):
        start, end = page_id * 50, (page_id + 1) * 50
        args = (urls, page_id, base_wd, start, end)
        pool.apply(get_html, args)
    # close the pool
    pool.close()