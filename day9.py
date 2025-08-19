# import library yang dibutuhkan
import mysql.connector
import pandas as pd

# tuliskan credential database untuk create connection
mydb = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    passwd = 'admin',
    database = 'employees'
)

# create access to DB
mycursor = mydb.cursor()
# tulis query SQL
query = '''select * from employees'''
# mengeksekusi query
mycursor.execute(query)
# gather all result
result = mycursor.fetchall()

# menampilkan hasil dalam bentuk table dengan pandas
df = pd.DataFrame(result, columns=mycursor.column_names)
print(df.head(5))