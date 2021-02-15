from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
lr_model_load = joblib.load("./models/lr_model.pkl")
tfidf_model_load = joblib.load("./models/tfidf_model.pkl")

# open a file, where you stored the pickled data
file = open("./models/reviews_text.pkl", "rb")
# load information to that file
reviews_data = pickle.load(file)
# close the file
file.close()

# open a file, where you stored the pickled data
file = open("./models/product_recommend.pkl", "rb")
# load information to that file
product_data = pickle.load(file)
# close the file
file.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():

    if (request.method == 'POST'):
        int_feature = request.form.get("username")
        user_recom = product_data[product_data['user_name'] == int_feature]

        df = reviews_data.join(user_recom.set_index('product_id'), on =['id'], how="inner")
        X_test_user = df['reviews_text']
        X_test_user = tfidf_model_load.transform(X_test_user)
        y_pred_user = lr_model_load.predict(X_test_user)
        user_sentiment = pd.DataFrame(data=y_pred_user, columns=['user_sentiment'],
                                       index=df.index.copy())
        fnl_result = pd.merge(df, user_sentiment, how='left', left_index=True, right_index=True)
        new_df = fnl_result.groupby('name')['user_sentiment'].mean()

        output = new_df.sort_values(ascending=False)[:5].reset_index()

        return render_template('index.html', prediction_text1='{}'.format(output.iloc[0]['name']),
                               prediction_text2='{}'.format(output.iloc[1]['name']),
                               prediction_text3='{}'.format(output.iloc[2]['name']),
                               prediction_text4='{}'.format(output.iloc[3]['name']),
                               prediction_text5='{}'.format(output.iloc[4]['name']),)
        # return "User Name is:"+int_feature
    else :
        return render_template('index.html')

@app.route("/predict_api", methods = ['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(lr_model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)