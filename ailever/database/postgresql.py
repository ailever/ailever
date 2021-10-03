from .__db_access_definition import DatabaseAccessObject

import os
import pandas as pd

class PostgreSQL(DatabaseAccessObject):
    def __init__(self, verbose=False):
        if verbose:
            self.installation_guide()

    def installation_guide(self):
        manual_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'installation_postgresql')
        with open(manual_path, 'r') as file:
            installation = file.read()
        print(installation)

    def meta_information(self):
        pass

    def connection(self):
        pass

    def execute(self):
        pass
