
def helper(what):
    if what == "server":
        print(server)
    elif what == "graph":
        print(graph)

server = """
jupyter lab &
python -m visdom.server &
rstudio-server start
service postgresql start
"""

graph = """
import matplotlib as mpl
mpl.get_cachedir()

# Korean Font
- Download1 : sudo apt-get install -y fonts-nanum fonts-nanum-coding fonts-nanum-extra
- Downlaod2 : https://hangeul.naver.com/font
- Upload : site-packages/matplotlib/mpl-data/fonts/ttf
- Check : head ~/.cache/matplotlib/fontList.json
"""
