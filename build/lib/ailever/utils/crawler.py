import requests
from bs4 import BeautifulSoup

def crawl(url, attr):
	headers = {'User-Agent': 'Mozilla/5.0'}
	response = requests.get(url, headers=headers)
	soup = BeautifulSoup(response.text)
	entities = soup.find_all(attr)
	return entities
