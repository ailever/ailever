import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def korean():
    for font in fm.fontManager.ttflist:
        if font.name == 'NanumBarunGothic':
            plt.rcParams["font.family"] = font.name
            break
