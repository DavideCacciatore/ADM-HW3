import multiprocessing as mp
import requests
import bs4
import time


def extract_urls(index, session):
    print("Executing task index:", index)
    sub_result = list()
    for num_page in range(index, index + 25):
        url = f"https://myanimelist.net/topanime.php?limit={num_page}"
        page = session.get(url)
        # Parsing the page
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        tag = soup.find_all('a', {'class': 'hoverinfo_trigger fl-l ml12 mr8'})
        for e in tag:
            sub_result.append(e.get("href"))
    #print(sub_result[0])
    return sub_result


def collect_result(result):
    global results
    results += result
    with open('urls.txt', 'w') as file:
        for url in results:
            file.write(str((url).encode("utf-8"))+"\n")


def parallelize_extraction():
    # 4 cores --> 16*4 tasks --> 25 pages per core per 4 times
    for starting_index in range(0, 400, 25):
        # print("Queuing task index:", starting_index)
        session = requests.Session()
        pool.apply_async(extract_urls, args=(starting_index, session), callback=collect_result)
    pool.close()
    pool.join()


if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())
    results = []
    start_time = time.time()
    print("Starting...")
    parallelize_extraction()
    print("--- %s seconds ---" % (time.time() - start_time))