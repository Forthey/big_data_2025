from bs4 import BeautifulSoup
import csv
import requests
from tqdm import tqdm


def parse_page(soup: BeautifulSoup) -> dict[int, float]:
    result: dict[int, float] = {}

    left_talbe = soup.find("div", class_="chronicle-table-left-column").find("table")
    right_table = soup.find("div", class_="chronicle-table").find("table")

    if None in (left_talbe, right_table):
        raise Exception("Parser miss table")
    
    left_tr_list = left_talbe.find_all("tr")[1:]
    right_tr_list = right_table.find_all("tr")[1:]

    assert len(left_tr_list) == len(right_tr_list)

    for index in tqdm(range(len(left_tr_list))):
        year = int(left_tr_list[index].find("td").get_text(strip=True))
        temp = float(right_tr_list[index].find_all("td")[12].get_text(strip=True))

        result[year] = temp

    return result


def get_data(url: str) -> dict[str, float]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
        
    return parse_page(BeautifulSoup(response.text, "html.parser"))


def save_data_as_csv(data: dict[int, str], filename: str):
    with open(filename, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["year", "temp"])
        for key, value in data.items():
            writer.writerow([key, value])


def main():
    baseurl: str = "http://www.pogodaiklimat.ru"

    print(f"Парсер сайта: {baseurl}")
    id:     int = int(input("Введите id города: "))

    data = get_data(f"{baseurl}/history/{id}.htm")

    save_data_as_csv(data, f"weather_{id}.csv")


if __name__ == "__main__":
    main()
