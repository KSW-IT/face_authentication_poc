
from datetime import datetime, timedelta

import sqlite3
import logging

DATABASE = "faces.db"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                embedding BLOB  NULL
            )
        ''')
        conn.commit()
#Deprecated
def save_user(name, email, embedding):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("INSERT INTO users (name, email, embedding) VALUES (?, ?, ?)", (name, email,str(embedding)))
        conn.commit()

def save_user2(email, embedding):# this function is new function to update face embedding in the existing user
     with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        # Check if the email already exists
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Update the existing user's embedding
            cursor.execute("UPDATE users SET embedding = ? WHERE email = ?", ( email,str(embedding)))
            logging.info("Face is updated in the existing user")
        else:
            cursor.execute("INSERT INTO users (email, embedding) VALUES (?, ?)", (email,str(embedding) ))
            logging.info("New user has created and face is registered")
        conn.commit()
#deprecated
def create_user(name,email,password):
    password_expiry = (datetime.now() + timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
   # password_expiry = (datetime.utcnow() + timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("INSERT INTO users (name, email, password,pass_expiry) VALUES (?, ?, ?, ?)", (name, email,password,password_expiry))
#deprecated
def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, embedding FROM users")
        return cursor.fetchall()
def get_user(email):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email FROM users where email = ?",(email,))
        #logging.info(f" user from getUser(email): {cursor.fetchall()}") 
        return cursor.fetchall()
    
def get_user2(email):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email , embedding FROM users where email = ?",(email,))
        
        #logging.info(f" user from getUser(email): {cursor.fetchall()}") 
        return cursor.fetchall()
    
def update_password(email,newPassword):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        # Check if the email already exists
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            password_expiry = (datetime.now() + timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("UPDATE users SET password = ? ,pass_expiry = ? WHERE email = ?", ( newPassword,password_expiry,email))
        else:
           logging.info(f" NO user found") 
        
        conn.commit()

    

