"""SQLite database for student sessions"""
import sqlite3
import os
from datetime import datetime


class Database:
    def __init__(self, db_path='data/cognitive_load.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE, created_date TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER, session_date TEXT,
            avg_cognitive_load REAL, accuracy REAL,
            adhd_risk REAL, dyslexia_risk REAL, details TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER, question TEXT, question_type TEXT,
            expected_load INTEGER, predicted_load INTEGER,
            response_time REAL, is_correct BOOLEAN)''')
        conn.commit()
        conn.close()

    def get_or_create_student(self, name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM students WHERE name = ?", (name,))
        row = c.fetchone()
        if row:
            sid = row[0]
        else:
            c.execute("INSERT INTO students (name, created_date) VALUES (?, ?)",
                      (name, datetime.now().isoformat()))
            sid = c.lastrowid
        conn.commit()
        conn.close()
        return sid

    def save_session(self, student_id, avg_load, accuracy, adhd_risk, dyslexia_risk):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO sessions
            (student_id, session_date, avg_cognitive_load, accuracy, adhd_risk, dyslexia_risk, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (student_id, datetime.now().isoformat(), avg_load, accuracy,
                   adhd_risk, dyslexia_risk, ''))
        sid = c.lastrowid
        conn.commit()
        conn.close()
        return sid

    def save_response(self, session_id, question, q_type, expected, predicted, rt, correct):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''INSERT INTO responses
            (session_id, question, question_type, expected_load, predicted_load, response_time, is_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (session_id, question, q_type, expected, predicted, rt, correct))
        conn.commit()
        conn.close()

    def get_student_sessions(self, name):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute('''SELECT s.id, s.session_date, s.avg_cognitive_load,
            s.accuracy, s.adhd_risk, s.dyslexia_risk
            FROM sessions s JOIN students st ON s.student_id = st.id
            WHERE st.name = ? ORDER BY s.session_date DESC''', (name,)).fetchall()
        conn.close()
        return rows

    def get_all_students(self):
        conn = sqlite3.connect(self.db_path)
        names = [r[0] for r in conn.execute("SELECT name FROM students ORDER BY name")]
        conn.close()
        return names