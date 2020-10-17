from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from .datasetdescription import DatasetDescription

class AileverAppEyes(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(AileverAppEyes, self).__init__(*args, **kwargs)

def run():
    app = QApplication(sys.argv)
    form = AileverAppEyes()
    form.show()
    app.exec_()
    print(f'[AILEVER] The Eyes is sucessfully executed!')

if __name__ == '__main__':
    run()
