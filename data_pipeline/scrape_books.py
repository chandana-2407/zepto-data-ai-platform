import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import sqlite3

BASE_URL = "https://books.toscrape.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Collect books from multiple categories
categories = {
    "Travel": "travel_2",
    "Mystery": "mystery_3",
    "Fiction": "fiction_10",
    "Romance": "romance_8",
    "Science": "science-fiction_16"
}

book_data = []

# -----------------------------
# SCRAPING
# -----------------------------

for category_name, category_path in categories.items():

    print(f"\nScraping category: {category_name}")

    url = f"{BASE_URL}catalogue/category/books/{category_path}/index.html"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Could not access {category_name}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.select("article.product_pod")

    print("Books found:", len(books))

    for book in books:

        title = book.h3.a["title"]

        price = book.select_one(".price_color").text.strip()

        rating = book.select_one(".star-rating")["class"][1]

        availability = book.select_one(".availability").text.strip()

        book_data.append({
            "title": title,
            "price_gbp": price,
            "rating": rating,
            "availability": availability,
            "category": category_name
        })

    time.sleep(1)


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

df = pd.DataFrame(book_data)

print("\n================================")
print("TOTAL BOOKS COLLECTED:", len(df))
print("================================")

print("\nBefore Cleaning:")
print(df.head())


# -----------------------------
# CLEAN PRICE
# -----------------------------

df["price_gbp"] = (
    df["price_gbp"]
    .str.replace(r"[^0-9.]","",regex=True)
    .astype(float)
)


# -----------------------------
# CLEAN RATING
# -----------------------------

rating_mapping = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating"].map(rating_mapping)


# -----------------------------
# CLEAN AVAILABILITY
# -----------------------------

df["in_stock"] = df["availability"].str.contains(
    "In stock",
    case=False,
    na=False
)

# Remove old availability column
df.drop(columns=["availability"], inplace=True)


# -----------------------------
# GBP TO INR CONVERSION
# -----------------------------

GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR


# -----------------------------
# FINAL COLUMN ORDER
# -----------------------------

df = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category"
    ]
]


# -----------------------------
# DISPLAY CLEAN DATA
# -----------------------------

print("\n================================")
print("CLEANED DATA")
print("================================")

print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nDataFrame Shape:")
print(df.shape)

# -----------------------------
# CREATE SQLITE DATABASE
# -----------------------------

import sqlite3

conn = sqlite3.connect("books.db")

df.to_sql(
    "books",
    conn,
    if_exists="replace",
    index=False
)

print("\nSQLite database created successfully!")
print("Books table created successfully!")

# Check data from SQLite
query = "SELECT * FROM books LIMIT 5"

sql_result = pd.read_sql_query(query, conn)

print("\nData from SQLite:")
print(sql_result)

conn.close()

# -----------------------------
# CREATE SQLITE DATABASE
# -----------------------------

conn = sqlite3.connect("books.db")

df.to_sql(
    "books",
    conn,
    if_exists="replace",
    index=False
)

print("\nSQLite database created successfully!")
print("Table 'books' created successfully.")

# -----------------------------
# SQL QUERIES
# -----------------------------

print("\n===== SQL QUERY 1: Total Books =====")

query1 = """
SELECT COUNT(*) AS total_books
FROM books
"""

result1 = pd.read_sql_query(query1, conn)
print(result1)


print("\n===== SQL QUERY 2: Books by Category =====")

query2 = """
SELECT category, COUNT(*) AS book_count
FROM books
GROUP BY category
ORDER BY book_count DESC
"""

result2 = pd.read_sql_query(query2, conn)
print(result2)


print("\n===== SQL QUERY 3: Average Price =====")

query3 = """
SELECT AVG(price_gbp) AS average_price_gbp
FROM books
"""

result3 = pd.read_sql_query(query3, conn)
print(result3)


print("\n===== SQL QUERY 4: Books with Rating 4 or Higher =====")

query4 = """
SELECT title, rating, price_gbp, category
FROM books
WHERE rating >= 4
"""

result4 = pd.read_sql_query(query4, conn)
print(result4)


print("\n===== SQL QUERY 5: Most Expensive Books =====")

query5 = """
SELECT title, price_gbp, category
FROM books
ORDER BY price_gbp DESC
LIMIT 5
"""

result5 = pd.read_sql_query(query5, conn)
print(result5)


# ==========================================
# CREATE RELATED TABLES: CATEGORIES + BOOKS
# ==========================================

print("\nCreating related tables...")

# Create categories table
conn.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE
)
""")

# Insert unique categories
for category in df["category"].unique():
    conn.execute(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        (category,)
    )

# Create new books table with Primary Key and Foreign Key
conn.execute("DROP TABLE IF EXISTS books")

conn.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

# Insert books with category_id
for _, row in df.iterrows():

    category_id = conn.execute(
        "SELECT category_id FROM categories WHERE category_name = ?",
        (row["category"],)
    ).fetchone()[0]

    conn.execute("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["price_gbp"],
        row["price_inr"],
        row["rating"],
        row["in_stock"],
        category_id
    ))

conn.commit()

print("Categories table created successfully!")
print("Books table created with Primary Key + Foreign Key!")

# ==========================================
# SQL JOIN
# ==========================================

print("\n===== SQL JOIN: Books with Categories =====")

join_query = """
SELECT
    b.book_id,
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books b
JOIN categories c
ON b.category_id = c.category_id
LIMIT 10
"""

join_result = pd.read_sql_query(join_query, conn)

print(join_result)

# ==========================================
# PANDAS MERGE
# ==========================================

print("\n===== Pandas Merge =====")

books_df = pd.read_sql_query(
    "SELECT * FROM books",
    conn
)

categories_df = pd.read_sql_query(
    "SELECT * FROM categories",
    conn
)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

print(merged_df.head(10))

conn.close()