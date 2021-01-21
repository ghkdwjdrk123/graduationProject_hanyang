import multiprocessing
import time
import requests
from bs4 import BeautifulSoup as bs
from multiprocessing import Manager

def get_links():
    req = requests.get('https://naver.com')
    html = req.text
    soup = bs(html, 'html.parser')
    my_titles = soup.select(
            'li > a'
            )
    data = []

    for title in my_titles:
        data.append(title.get('href'))
    return data


def function(result_list, link):
    req = requests.get(link)
    time.sleep(1)
    html = req.text
    soup = bs(html, 'html.parser')
    head = soup.select('h1')
    if len(head) != 0:
        result_list.append(head[0].text.replace('\n', ''))

if __name__ == '__main__':
    start = time.time()
    pool = multiprocessing.Pool(8)
    m = Manager()
    result_list = m.list()
    all_links = [x for x in get_links() if x.startswith('http')]

    pool.starmap(function, [(result_list, link) for link in all_links])
    pool.close()
    pool.join()
    print("%s"%(time.time() - start))
