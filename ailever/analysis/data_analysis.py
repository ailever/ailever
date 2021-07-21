import pandas as pd

class Table:
    def __init__(self, frame):
        self.frame = frame

    def CountsByInstance(self):
        Base = pd.DataFrame(columns=['Column', 'Instance', 'Count'])
        for column in self.frame.columns:
            base = self.frame[column].value_counts().reset_index().rename(columns={'index':'Instance', column:'Count'})
            base.insert(0, 'Column', column)
            Base = Base.append(base)
        return Base

