from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

lg = pickle.load(open('logisticRegression.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    prediction = 5
    if request.method == 'POST':
        review = request.form['user_input']
        data = [review]
        vect = tfidf.transform(data).toarray()
        prediction = lg.predict(vect)[0]
    return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run()