import streamlit as st
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import pickle

#Load saved df and files
df1 = pd.read_csv("final_db.csv")
tfidf = pickle.load(open("tfidf.pkl",'rb'))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl",'rb'))

def seach_based_recommendation(user_book_title, df, tfidf, tfidf_matrix, top_n=10):
    #covert user input to vector
    query_vector = tfidf.transform([user_book_title])

    #match userinput and tfidf dataset
    cosin_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()

    #[0:0.90,1:0.4,3:0.89]
    similarity_score = list(enumerate(cosin_sim))
    similarity_score = sorted(similarity_score, key=lambda x:x[1], reverse=True)

    #pic top 10 highest similariy score
    similarity_score = similarity_score[1:top_n+1]

    #book indexes
    book_indices = [i[0] for i in similarity_score]

    #return the relevant columns for the book recommendation
    return df[['Book-Title','Book-Rating','Image-URL-M']].iloc[book_indices].reset_index(drop=True)


#UI Design using streamlit
st.title("Search Based Book Recommendation System")
book_to_search = st.text_input("Enter a book title to search for recommendation")
if book_to_search:
    recommendations = seach_based_recommendation(book_to_search,df1, tfidf, tfidf_matrix)
    if not recommendations.empty:
        cols = st.columns(5)
        for i,row in recommendations.iterrows():
            col = cols[i%5]
            col.image(row['Image-URL-M'], width=100)
            col.markdown(f"**{row['Book-Title']}**")