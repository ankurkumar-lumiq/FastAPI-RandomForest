# 1. Library Imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Json
import joblib
import pandas as pd
from Random_Forest_train import preprocess_text


# 2. Create the my app object
myapp = FastAPI()


# 3. Loading the model.
rf_model = joblib.load('Random_Forest_trained.joblib')
tfidf = joblib.load('tfidf.joblib')
sc = joblib.load('scaler.joblib')
norm = joblib.load('normalizer.joblib')
# preprocess = joblib.load('preprocess.joblib')


# 4. Index route , opens automatically on http://127.0.0.1:8000
@myapp.get('/')
def index():
    return {'message':'Hello'}


# 5. class Taking review
class review_data(BaseModel):
    review: str


# 6. Expose the Predictions
@myapp.post('/predict')
def predict_rating(data:review_data):
    review = data.review

    # Prepairing data for predictions
    cleaned_text_review = preprocess_text(review)

    # applying TF-IDF
    tfidf_matrix = tfidf.transform([cleaned_text_review]).toarray()
    tfidf_review = pd.DataFrame(data = tfidf_matrix,columns = tfidf.get_feature_names_out())
    tfidf_review['word_count'] = [len(cleaned_text_review.split())]
    tfidf_review['char_count'] = len(cleaned_text_review)

    # applying scaling
    sc_review = sc.transform(tfidf_review)
    st_review = pd.DataFrame(sc_review,columns= tfidf_review.columns)

    # applying normalization
    nor_review = norm.transform(st_review)
    final_review = pd.DataFrame(nor_review, columns= st_review.columns)
    
    #prediciting
    prediction = rf_model.predict(final_review)
    
    return {"Prediction that customer would rate":int(prediction[0])+1}


# 7. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(myapp,host='127.0.0.1',port=8000)
