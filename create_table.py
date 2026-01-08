from connection_manager import db_connection

sql_create = """
    DROP TABLE IF EXISTS samples_database.{table_name};
    CREATE TABLE IF NOT EXISTS samples_database.{table_name}   
    (
        id integer NOT NULL GENERATED ALWAYS AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 2147483647 CACHE 1),
        scan_number integer,
        channel_data BYTEA
    );

    ALTER TABLE IF EXISTS samples_database.{table_name}
        OWNER TO {db_user};
    """
   
def create_raw_table():
    with db_connection() as conn:
        curr = conn.cursor()
        curr.execute(sql_create.format(table_name = "numpy_raw_data", db_user = "postgres"))
        conn.commit()