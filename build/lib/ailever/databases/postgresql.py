from .__db_access_definition import DatabaseAccessObject


class PostgreSQL(DatabaseAccessObject):
    def __init__(self, verbose=False):
        if verbose:
            self.installation_guide()

    def installation_guide(self):
        with open('installation_postgresql', 'r') as file:
            installation = file.read()
        print(installation)

    def meta_information(self):
        pass

    def connection(self):
        pass

    def execute(self):
        pass
