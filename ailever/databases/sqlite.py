from .__db_access_definition import DatabaseAccessObject


class SQLite(DatabaseAccessObject):
    def __init__(self, verbose=False):
        if verbose:
            self.installation_guide()

    def installation_guide(self):
        pass

    def connection(self):
        pass

    def execute(self):
        pass
