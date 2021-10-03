from .__db_access_definition import DatabaseAccessObject


class PostgreSQL(DatabaseAccessObject):
    def __init__(self, verbose=False):
        if verbose:
            self.installation_guide()

    def installation_guide(self):
        print("""
$ sudo apt update
$ sudo apt install postgresql postgresql-contrib
$ sudo service --status-all
$ sudo service postgresql start
        """)

    def connection(self):
        pass

    def execute(self):
        pass
