import multiprocessing as mp
import requests
import bs4


def extract_urls(index, session):
    print("Executing task index:", index)
    sub_result = list()
    for num_page in range(index, index + 25):
        url = f"https://myanimelist.net/topanime.php?limit={num_page*50}"
        page = session.get(url)
        # Parsing the page
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        tag = soup.find_all('a', {'class': 'hoverinfo_trigger fl-l ml12 mr8'})
        for e in tag:
            sub_result.append(e.get("href"))
    return sub_result


def parallelize_extraction():
    pool = mp.Pool(mp.cpu_count())
    # 4 cores --> 16*4 tasks --> 25 pages per core per 4 times
    results = list()
    for starting_index in range(0, 400, 25):
        session = requests.Session()
        results += pool.apply(extract_urls, args=(starting_index, session))
    pool.close()
    with open('url_of_interest.txt', 'w', encoding="utf-8") as file:
        for url in results:
            file.write(url + "\n")




