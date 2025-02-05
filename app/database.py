import sqlite3
import logging

DATABASE = "faces.db"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        conn.commit()

def save_user(name, email, embedding):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("INSERT INTO users (name, email, embedding) VALUES (?, ?, ?)", (name, email,str(embedding)))
        conn.commit()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name ,email, embedding FROM users")
        return cursor.fetchall()
def get_user(email):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email FROM users where email = ?",(email,))
        #logging.info(f" user from getUser(email): {cursor.fetchall()}") 
        return cursor.fetchall()
    

