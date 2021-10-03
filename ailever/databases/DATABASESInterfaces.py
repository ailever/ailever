

def DB(database, verbose=False):
    database = database.lower()
    if database == 'oracle':
        from .oracle import Oracle
        db_aco = Oracle(verbose)
    elif database == 'postgresql':
        from .postgresql import PostgreSQL
        db_aco = PostgreSQL(verbose)
    elif database == 'mysql':
        from .mysql import MySQL
        db_aco = MySQL(verbose)
    elif database == 'sqlite':
        from .sqlite import SQLite
        db_aco = SQLite(verbose)
    else:
        return None
    return db_aco
        
