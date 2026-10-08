import sqlite3
from datetime import datetime

DB_PATH = "data/patients.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Patient registration table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            contact TEXT,
            blood_group TEXT,
            medical_notes TEXT,
            registered_on TIMESTAMP
        )
    """)
    
    # Prediction history table linked to patient
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            disease TEXT,
            risk_score REAL,
            result TEXT,
            assessed_on TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)
    
    conn.commit()
    conn.close()

def register_patient(patient_id, name, age, gender, contact, blood_group, notes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (patient_id, full_name, age, gender, contact, blood_group, medical_notes, registered_on)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, name, age, gender, contact, blood_group, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id, full_name, age, gender, contact, blood_group FROM patients ORDER BY registered_on DESC")
    records = cursor.fetchall()
    conn.close()
    return records