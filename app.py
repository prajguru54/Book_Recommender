from flask.helpers import url_for
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

df = pd.read_csv('books.csv', error_bad_lines=False)
df.columns = [c.strip() for c in list(df.columns)] # num_pages column has some leading spaces
df_original = df.copy()
df_original.title = [i.lower() for i in df_original.title.values]


# Compute the similarity matrix
def get_similarity_matrix():  
 
    features = ['num_pages', 'ratings_count', 'text_reviews_count', 'publisher', 'average_rating']
    df_truncated = df.loc[:,features]

    numeric_df = df_truncated.drop('publisher', axis = 1)
    catagorical_df = df_truncated['publisher']

    from sklearn.feature_extraction.text import TfidfVectorizer
    tf_df = TfidfVectorizer()
    tfidf_transformed_features = tf_df.fit_transform(catagorical_df)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    scaled_numeric_features = sc.fit_transform(numeric_df)
    scaled_nmeric_df = pd.DataFrame(scaled_numeric_features, columns=numeric_df.columns)
    df_prepared = pd.concat([scaled_nmeric_df, pd.DataFrame(tfidf_transformed_features.toarray())], axis = 1)

    from sklearn.metrics.pairwise import cosine_similarity
    cm = cosine_similarity(df_prepared)

    return cm

# Calculate the similarity of the provided book with other books
def calculate_similarity_scores(book_name, cm):

    matched_row = df_original[df_original.title == book_name]
    similarity_scores = cm[matched_row.index.values[0], :] 
    related_books = []
    for i in range(len(similarity_scores)):
        if (i, similarity_scores[i]) not in related_books:
            related_books.append((i, similarity_scores[i]))
            
    top_five_scores = sorted(related_books, key=lambda x:x[1], reverse=True)[1:6]
    top_five_indices = [top_five_scores[i][0] for i in range(len(top_five_scores))]
    return top_five_indices

# Recommend the top five books which are most similar to the the given book
def recommend(top_five_indices):
    top_five_recommendations = df_original.iloc[top_five_indices, :]['title'].values
    print('\n--------------Top 5 Recommendations are:--------------')
    for index, reco in enumerate(top_five_recommendations):
        print(f'{index+1}: {reco}')
    return top_five_recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST': 
        book_name = request.form['book_name'].lower()
        cm = get_similarity_matrix()       
        try:
            top_five_indices = calculate_similarity_scores(book_name, cm)
            top_five_recommendations = recommend(top_five_indices)
            return render_template('results.html',  b1 = top_five_recommendations[0], 
                                                b2 = top_five_recommendations[1], 
                                                b3 = top_five_recommendations[2], 
                                                b4 = top_five_recommendations[3], 
                                                b5 = top_five_recommendations[4])
        except:
            error_message = "Oppps...Coudn't find the book in Database."
            return render_template('error_page.html',  message = error_message)
        
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)


