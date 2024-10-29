import sqlite3

# Membuka koneksi ke database
conn = sqlite3.connect('attendance.db')

# Membuat cursor untuk mengeksekusi query
cursor = conn.cursor()

# Contoh menampilkan tabel-tabel yang ada
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tabel dalam database:", tables)

# Jangan lupa menutup koneksi
conn.close()
