from urllib.request import urlretrieve

def loader(url='https://raw.githubusercontent.com/ailever/ailever/master/API.md'):
    File, Header = urlretrieve(url, f'./{url}')
    print(Header)


loader()
