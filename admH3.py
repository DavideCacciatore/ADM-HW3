import glob
import multiprocessing as mp
import os
import requests
import bs4
import time


def extract_urls(index, session):
    print("Executing task index:", index)
    sub_result = list()
    for num_page in range(index, index + 25):
        url = f"https://myanimelist.net/topanime.php?limit={num_page*50}"
        #print(url)
        page = session.get(url)
        # Parsing the page
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        tag = soup.find_all('a', {'class': 'hoverinfo_trigger fl-l ml12 mr8'})
        for e in tag:
            sub_result.append(e.get("href"))
    #print(sub_result[0])
    return sub_result


'''def collect_result(result):
    global results
    results = result
    with open('urls.txt', 'w', encoding="utf-8") as file:
        for url in results:
            file.write(url +"\n")'''


def parallelize_extraction():
    # 4 cores --> 16*4 tasks --> 25 pages per core per 4 times
    results = list()
    for starting_index in range(0, 400, 25):
        # print("Queuing task index:", starting_index)
        session = requests.Session()
        results += pool.apply(extract_urls, args=(starting_index, session)) #, callback=collect_result)
    pool.close()
    #pool.join()
    with open('url_of_interest.txt', 'w', encoding="utf-8") as file:
        for url in results:
            file.write(url + "\n")


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


def parse_anime(starting_page, path):
    for i in range(starting_page, starting_page + 28):
        dir_name = path + "\data\page_" + i
        for file in glob.iglob(r"" + {dir_name} + '\*.html'):
            with open(file, encoding="utf-8") as fp:
                soup = bs4.BeautifulSoup(fp, "html.parser")
                # animeTitle = soup.head.title.split(" -")[0]
                animeInfo = soup.find("div", {"class": "spaceit_pad"}, string="Type:")
                # print(animeTitle)
                print(animeInfo)


def parallelize_parsing(path):
    # 28 tasks : 4 pages per task
    page = 270
    for r in range(0, 112, 28):
        pool.apply_async(parse_anime, args=(page + r, path))  # , callback=collect_result)
    pool.close()
    pool.join()


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    start_time = time.time()
    print("Starting...")
    #parallelize_extraction()
    #pool = mp.Pool(mp.cpu_count())
    #get_content()
    #pool = mp.Pool(mp.cpu_count())
    parallelize_parsing(os.getcwd())

    print("--- %s seconds ---" % (time.time() - start_time))