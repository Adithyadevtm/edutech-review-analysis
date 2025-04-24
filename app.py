from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def home():
    return "API is working! Send a POST request to /predict with 'text'!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
