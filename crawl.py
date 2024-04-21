import requests
from bs4 import BeautifulSoup


def crawl(url: str):
    r = requests.get(url)
    # r.encoding = r.apparent_encoding
    html_content = r.text
    soup = BeautifulSoup(html_content)
    bookname = str(soup.title.get_text()).replace("博客來-", "")
    cover = soup.find("meta", attrs={"property": "og:image"}).get(
        "content").split("&")[0]
    description = soup.find(
        "div", attrs={"class": "bd"}).get_text().replace("\n", "").replace("\xa0", "")
    return {"bookname": bookname, "cover": cover, "description": description}


if __name__ == "__main__":
    url = "https://www.books.com.tw/products/0010986499?loc=P_0325__1001"
    print(crawl(url))
