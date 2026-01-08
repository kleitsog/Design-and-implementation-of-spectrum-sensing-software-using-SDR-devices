import contextlib
import psycopg2

db_user = "postgres"
db_passwd = "postgres"
db_port = "5432"
db_name = "samples_database"

connection_string = f"postgresql://{db_user}:{db_passwd}@localhost:{db_port}/{db_name}"

@contextlib.contextmanager
def db_connection():
    conn = psycopg2.connect(connection_string)

    try:
        yield conn 
    except Exception as e:
        print(e)
    finally:
        conn.close()
