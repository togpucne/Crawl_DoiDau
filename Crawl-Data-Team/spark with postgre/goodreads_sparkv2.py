from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_date, year, count, avg, monotonically_increasing_id, when

# Khởi tạo phiên Spark với MongoDB và PostgreSQL
spark = SparkSession.builder \
    .appName("Goodreads Spark with MongoDB and PostgreSQL") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.postgresql:postgresql:42.7.4") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/goodreads_db.books") \
    .getOrCreate()

# Thiết lập chính sách phân tích cú pháp thời gian thành LEGACY để xử lý ngày tháng cũ
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Cài đặt mức độ log
spark.sparkContext.setLogLevel("INFO")

# Đọc dữ liệu từ MongoDB với một số tùy chọn tối ưu hóa (nếu dataset lớn)
df = spark.read \
    .format("mongo") \
    .option("uri", "mongodb://localhost:27017/goodreads_db.books") \
    .option("partitioner", "MongoPaginateBySizePartitioner") \
    .option("partitionSizeMB", "64") \
    .load()

# Hiển thị dữ liệu từ MongoDB ban đầu
df.show(5)
df.printSchema()

# Bước 1: Xử lý cột publish_date - Loại bỏ tiền tố "First published" và chuyển đổi thành kiểu ngày tháng
df = df.withColumn("cleaned_date", regexp_replace(col("Date"), "First published ", "")) \
       .withColumn("publish_date", to_date(col("cleaned_date"), "MMMM d, yyyy"))

# Bước 2: Xử lý dữ liệu có dạng "8,932,568" - Loại bỏ dấu phẩy
df = df.withColumn("Number of Ratings", regexp_replace(col("Number of Ratings"), ",", "")) \
       .withColumn("Reviews", regexp_replace(col("Reviews"), ",", "")) \
       .withColumn("Score", regexp_replace(col("Score"), ",", ""))

# Bước 3: Chuyển đổi kiểu dữ liệu sau khi loại bỏ dấu phẩy
df = df.withColumn("Pages", col("Pages").cast("int")) \
       .withColumn("Rating", col("Rating").cast("float")) \
       .withColumn("Number of Ratings", col("Number of Ratings").cast("int")) \
       .withColumn("Reviews", col("Reviews").cast("int")) \
       .withColumn("Score", col("Score").cast("int"))

# Bước 4: Xử lý dữ liệu null, nếu có giá trị null sẽ thay bằng giá trị mặc định
df = df.na.fill({
    "Pages": 0,
    "Rating": 0.0,
    "Number of Ratings": 0,
    "Reviews": 0,
    "Score": 0
})

# Tạo các bảng từ dữ liệu

# Bảng authors chứa thông tin về tác giả, sử dụng distinct để loại bỏ các giá trị trùng lặp
authors_df = df.select("Author").distinct(
).withColumnRenamed("Author", "author_name")

# Bảng books chứa thông tin về sách và ngày xuất bản
books_df = df.select("Title", "Author", "Pages", "Cover Type", "publish_date") \
             .withColumnRenamed("Title", "book_title") \
             .withColumnRenamed("Author", "author_name") \
             .withColumnRenamed("Pages", "num_pages") \
             .withColumnRenamed("Cover Type", "cover_type")

# Thêm cột year_published
books_df = books_df.withColumn("year_published", year(col("publish_date")))

# Thêm cột num_authors_books: Số lượng sách mỗi tác giả đã viết
author_book_count_df = books_df.groupBy("author_name").agg(
    count("book_title").alias("num_authors_books"))
books_df = books_df.join(author_book_count_df, "author_name", "left")

# Bảng ratings chứa thông tin về đánh giá
ratings_df = df.select("Title", "Rating", "Number of Ratings", "Reviews", "Score") \
               .withColumnRenamed("Title", "book_title") \
               .withColumnRenamed("Rating", "rating") \
               .withColumnRenamed("Number of Ratings", "num_ratings") \
               .withColumnRenamed("Reviews", "num_reviews") \
               .withColumnRenamed("Score", "score")

# Thêm cột rating_category
ratings_df = ratings_df.withColumn(
    "rating_category",
    when(col("rating") >= 4.5, "Excellent")
    .when((col("rating") >= 3.5) & (col("rating") < 4.5), "Good")
    .otherwise("Average")
)

# Thêm cột average_author_rating: Đánh giá trung bình của mỗi tác giả dựa trên các sách
author_avg_rating_df = ratings_df.join(books_df, "book_title", "inner") \
    .groupBy("author_name").agg(avg("rating").alias("average_author_rating"))

# Bảng genres chứa thông tin về thể loại
genres_df = df.select("Genres").distinct(
).withColumnRenamed("Genres", "genre_name")

# Thêm cột book_count_by_genre
genre_book_count_df = df.groupBy("Genres").agg(
    count("Title").alias("book_count_by_genre"))
genres_df = genres_df.join(genre_book_count_df.withColumnRenamed(
    "Genres", "genre_name"), "genre_name", "left")

# Bảng books_genres liên kết nhiều-nhiều
books_genres_df = df.select("Title", "Genres") \
    .withColumnRenamed("Title", "book_title") \
    .withColumnRenamed("Genres", "genre_name")

# Tạo khóa chính cho bảng books và authors
books_df = books_df.withColumn("book_id", monotonically_increasing_id())
authors_df = authors_df.withColumn("author_id", monotonically_increasing_id())

# Thêm khóa ngoại vào bảng ratings để kết nối với bảng books
ratings_df = ratings_df.join(books_df.select(
    "book_title", "book_id"), on="book_title", how="inner")

# Thêm khóa ngoại vào bảng books để kết nối với bảng authors
books_df = books_df.join(authors_df.select(
    "author_name", "author_id"), on="author_name", how="inner")

# Xử lý bảng books_genres (quan hệ nhiều-nhiều)
books_genres_df = books_genres_df.join(books_df.select("book_title", "book_id"), on="book_title", how="inner") \
    .join(genres_df.select("genre_name"), on="genre_name", how="inner") \
    .select("book_id", "genre_name")

# Loại bỏ các giá trị rỗng hoặc bằng 0
books_df = books_df.filter((books_df["num_pages"] > 0) & (
    books_df["publish_date"].isNotNull()))
ratings_df = ratings_df.filter(
    (ratings_df["rating"] > 0) & (ratings_df["num_ratings"] > 0))

# Sắp xếp lại các cột cho dễ đọc

# Bảng authors: Sắp xếp cột
authors_df = authors_df.select("author_id", "author_name")

# Bảng books: Sắp xếp cột
books_df = books_df.select("book_id", "book_title", "author_name", "num_pages",
                           "cover_type", "publish_date", "year_published", "num_authors_books")

# Bảng ratings: Sắp xếp cột
ratings_df = ratings_df.select("book_id", "book_title", "rating",
                               "rating_category", "num_ratings", "num_reviews", "score")

# Bảng genres: Sắp xếp cột
genres_df = genres_df.select("genre_name", "book_count_by_genre")

# Bảng books_genres: Sắp xếp cột
books_genres_df = books_genres_df.select("book_id", "genre_name")

# Kết nối tới PostgreSQL và lưu lại các bảng
jdbc_url = "jdbc:postgresql://localhost:5432/goodreads_booksv2"
connection_properties = {
    "user": "postgres",
    "password": "fafa123123haha.",
    "driver": "org.postgresql.Driver"
}


def write_to_postgres(df, table_name):
    try:
        df.write.jdbc(
            url=jdbc_url,
            table=table_name,
            mode="overwrite",
            properties=connection_properties
        )
        print(f"Đã ghi thành công bảng {table_name} vào PostgreSQL!")
    except Exception as e:
        print(f"Lỗi khi ghi bảng {table_name} vào PostgreSQL: {e}")


# Lưu các bảng vào PostgreSQL
write_to_postgres(authors_df, "authors")
write_to_postgres(books_df, "books")
write_to_postgres(ratings_df, "ratings")
write_to_postgres(genres_df, "genres")
write_to_postgres(books_genres_df, "books_genres")
