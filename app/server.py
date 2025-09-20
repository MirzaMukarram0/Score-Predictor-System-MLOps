from flask import Flask, request, jsonify
import pickle
import os


MODEL_PATH = os.getenv("MODEL_PATH", "../datasets/student_model.pkl")


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


app = Flask(__name__)
model = load_model()  # load immediately at startup ✅


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Invalid input"}), 400

    features = data["features"]
    try:
        prediction = model.predict([features])
        return jsonify({"prediction": float(prediction[0])})  # unwrap to float ✅
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
