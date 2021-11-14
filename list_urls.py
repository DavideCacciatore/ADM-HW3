import multiprocessing as mp
import requests
import bs4

# task function to extract data - return a list of sub results
def extract_urls(index, session):
    print("Executing task index:", index)
    sub_result = list()
    # for each page in the pagination, extract the urls
    for num_page in range(index, index + 25):
        # connect to the page
        url = f"https://myanimelist.net/topanime.php?limit={num_page*50}"
        page = session.get(url)
        # parse the page
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        tag = soup.find_all('a', {'class': 'hoverinfo_trigger fl-l ml12 mr8'})
        # memorize urls in a list to be returned
        for e in tag:
            sub_result.append(e.get("href"))
    return sub_result


# function to handle the parallelization
def parallelize_extraction():
    # create a pool to be later populated and define a coherent number of tasks to exploit parallelization better
    pool = mp.Pool(mp.cpu_count())
    # 4 cores --> 16 tasks --> 25 pages per core (per 4 times)
    results = list()
    for starting_index in range(0, 400, 25):
        # define a connection session for each task
        session = requests.Session()
        # launch task and collect results
        results += pool.apply(extract_urls, args=(starting_index, session))
    # close the pool
    pool.close()
    # write the collected data on a stored file line by line
    with open('url_of_interest.txt', 'w', encoding="utf-8") as file:
        for url in results:
            file.write(url + "\n")




