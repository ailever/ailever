
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

# Fonts Path
Dwonlaod : https://hangeul.naver.com/font
Upload : site-packages/matplotlib/mpl-data/fonts/ttf
"""
