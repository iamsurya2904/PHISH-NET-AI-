from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
from Feature import FeatureExtraction

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load the pre-trained model from the pickle file
with open( 'pickle/model.pkl', "rb") as model_file:
    gbc = pickle.load(model_file)

# Initialize the Flask application
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the URL from the form input
        input_url = request.form.get("url")
        
        # Extract features from the URL
        feature_extractor = FeatureExtraction(input_url)
        feature_list = np.array(feature_extractor.getFeaturesList()).reshape(1, 30)
        
        # Make predictions using the loaded model
        prediction = gbc.predict(feature_list)[0]
        proba_phishing = gbc.predict_proba(feature_list)[0, 0]
        proba_non_phishing = gbc.predict_proba(feature_list)[0, 1]
        
        # Format the prediction result
        result_message = "It is {:.2f}% safe to go".format(proba_phishing * 100)
        
        # Render the result back to the HTML page
        return render_template('index.html', xx=round(proba_non_phishing, 2), url=input_url)
    
    # Render the index page with default values when method is GET
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    # Start the Flask development server
    app.run(debug=True)
