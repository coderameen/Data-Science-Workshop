#Load the Dataset
import pandas as pd
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
# print(books)
# print(ratings)
# print(ratings['Book-Rating'].value_counts())

#merge both datasets
df = pd.merge(books, ratings, on='ISBN', how='left')
# print(df.head())


#Data Cleaning
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())


#shape
print(df.shape)#(1031128, 10) 1M books
#But, I need only 20,000 books
df = df.sample(n=20000, random_state=42).reset_index(drop=True)
print(df.shape)#(20000,10)
#now save in new csv
df.to_csv("final_db.csv")


#Popularity Base Recommendation System
'''
1. Step 1:Group the dataset by Book-Title and calculate
 the average rating and the number of rating for
 each books
2. step 2: Sort the books based on either the average rating
or the number of rating(popularity)
3. step 3: Recommed books that have the highest popularity score.
'''


#step 1:
book_ratings = df.groupby('Book-Title').agg({'Book-Rating':['mean','count']}).reset_index()
print(book_ratings)

#step 2
book_ratings.columns = ['Book-Title','Avg-Rating','Num-Ratings']
book_ratings_sorted = book_ratings.sort_values(by=['Num-Ratings','Avg-Rating'],ascending=False)
top_books = book_ratings_sorted.head(10)

top_books_final = pd.merge(top_books, df[['Book-Title','Image-URL-M']], on='Book-Title',how='left')
print(top_books_final)

#here we have duplicates
top_books_final = top_books_final.drop_duplicates(subset=['Book-Title'])
print(top_books_final)

#Save this top best 10 books top_books_final.csv
top_books_final.to_csv("top_books_final.csv")