import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("final_db.csv")
# print(df.head())

#here we may of may not have duplicate values
df = df.drop_duplicates(subset=['Book-Title'])

#only i need book-title column
# print(df['Book-Title'])

#create tfidf object on book-title
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Book-Title'])
print(tfidf_matrix.toarray())

from sklearn.metrics.pairwise import cosine_similarity
'''
consine similarity multiple(*)/dot product with userinput and tfidf dataset
for providing to match/recommend the result
ex:
userinput
     s1 = "i love cat"
tfidf dataset
    s2 = "i love dog"
[10, 20, 30] . [10, 20,55] = 0.90(most higher matching item)

ex2: no i love in dataset
[10, 20, 30] . [10,40,50] = 0.20(lower not matching item)

'''

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


#Example Usage
book_to_search = "Pinocchio"
recommend_books = seach_based_recommendation(book_to_search,df, tfidf, tfidf_matrix)

print(recommend_books)

#pickle the tfidf and tfidf_matrix
#save tfidf and tfidf_matrix using pickle
import pickle
pickle.dump(tfidf, open("tfidf.pkl",'wb'))
pickle.dump(tfidf_matrix, open("tfidf_matrix.pkl",'wb'))



