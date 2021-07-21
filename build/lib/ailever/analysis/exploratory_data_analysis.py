import pandas as pd


class ExploratoryDataAnalysis:
    def __init__(self, frame):
        self.frame = frame

    def cleaning(self):
        pass

    def frequency(self):
        pass

    def transformation(self):
        pass

    def selection(self):
        pass

    def visualization(self):
        pass


class Counting:
    def __init__(self, frame, path='ExploratoryDataAnalysis'):
        self.frame = frame
        self.path = path

    def CountsByInstance(self, path=None, save=False):
        Base = pd.DataFrame(columns=['Column', 'Instance', 'Count'])
        for column in self.frame.columns:
            base = self.frame[column].value_counts().reset_index().rename(columns={'index':'Instance', column:'Count'})
            base.insert(0, 'Column', column)
            Base = Base.append(base)

    	if save:
            if not path:
                if not os.path.isdir(self.path):
                    os.mkdir(self.path)
                Base.to_csv(os.path.join(self.path, 'CountsByInstance.csv'))
            else:
                if not os.path.isdir(path):
                    os.mkdir(path)
                Base.to_csv(os.path.join(path, 'CountsByInstance.csv'))

        return Base
