
def helper(what):
    if what == "server":
        print(server)

server = """
jupyter lab &
python -m visdom.server &
rstudio-server start
service postgresql start
"""
