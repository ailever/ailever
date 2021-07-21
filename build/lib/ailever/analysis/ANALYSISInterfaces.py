from .data_analysis import Table

def EDA(frame):
    table = Table(frame)
    return table.CountsByInstance()
