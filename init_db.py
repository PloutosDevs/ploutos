import sqlite3
from config import DATA_PATH, PATH
import os

def initialize_database():
    conn = sqlite3.connect(os.path.join(DATA_PATH, "ploutos_data.db"))
    with open(os.path.join(PATH, 'sql' ,"init.sql"), "r") as f:
        conn.executescript(f.read())
    conn.close()

if __name__ == "__main__":
    initialize_database()