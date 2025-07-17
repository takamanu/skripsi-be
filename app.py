from flask import Flask, request, jsonify, flash, redirect, send_file
import mimetypes
from xg_boost import predict_diabetes

from xg_boost import process
from flask_cors import CORS
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key"
# agar tidak terjadi block antar be dan fe
CORS(app)


@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/start", methods=["POST"])
def start_process():
    filepath = "dataset/diabetes.csv"

    # Check if file exists
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found: dataset/diabetes.csv"}), 404

    data = pd.read_csv(filepath)

    # Call process from xg_boost.py
    (
        result1, result2, result3, result4, result5, result7, result8, result9, result10
    ) = process(filepath)

    return jsonify({
        "result1": result1.to_dict(),  # OR .values.tolist()
        "datasetBefore": result1.values.tolist(),
        "dataLabelEncode": result2.to_dict(),
        "accXGBoost": result4,
        "accXGBoostAndSmote": result3,
        "accXGBoostAndSmoteTomek": result5,
        "df_Smote": result7.to_dict(),
        "df_Adasyn": result8.to_dict(),
        "df_SvmSmote": result9,
        "datasetAfter": result10.values.tolist()
    })

@app.route("/images/<filename>", methods=["GET"])
def get_image(filename):
    file_path = f"images/{filename}"
    
    # Memeriksa apakah file eksis
    if not os.path.exists(file_path):
        return "File not found", 404

    # Mendapatkan tipe konten berdasarkan ekstensi file
    content_type, _ = mimetypes.guess_type(file_path)
    
    # Mengembalikan file dengan tipe konten yang benar
    return send_file(file_path, mimetype=content_type)



@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil JSON dari request body
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "Request body kosong atau bukan JSON"}), 400

        # Jalankan prediksi
        prediction = predict_diabetes(input_data)

        return jsonify({
            "input": input_data,
            "prediction": prediction,
            "description": "1 = Diabetes, 0 = Tidak Diabetes"
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except FileNotFoundError as fe:
        return jsonify({"error": str(fe)}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

