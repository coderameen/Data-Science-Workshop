import streamlit as st
import pandas as pd

#load the saved DataFrame
df = pd.read_csv("top_books_final.csv")
# print(df.head())

#Streamlit UI
st.title("Popularity Based Recommendation System")
st.subheader("Top 10 Popular Books")

#1 row: 5 books(cols) 0-4
col1, col2, col3, col4, col5 = st.columns(5)
for i, col in enumerate([col1,col2,col3,col4,col5]):
    if i < len(df):
        col.image(df.loc[i, 'Image-URL-M'])
        col.markdown(df.loc[i, 'Book-Title'])
        col.write(f"{df.loc[i,'Num-Ratings']}")



#2 row: 5-9
col6, col7, col8, col9,col10 = st.columns(5)

for i,col in enumerate([col6, col7, col8, col9,col10]):
    if i+5 < len(df):
        col.image(df.loc[i+5, 'Image-URL-M'])
        col.markdown(df.loc[i+5, 'Book-Title'])
        col.write(f"{df.loc[i+5,'Num-Ratings']}")

