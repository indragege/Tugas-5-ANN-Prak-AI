from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Pastikan folder static ada
os.makedirs('static', exist_ok=True)

# Global variables
model = None
data_tahunan = None
scaler = None

def load_model_and_data():
    global model, data_tahunan, scaler
    try:
        # Load model
        model = tf.keras.models.load_model('pneumonia_prediction_model.h5')
        
        # Load data
        df = pd.read_csv('dinkes-od_18513_jml_kasus_penyakit_pneumonia__kabupatenkota_v2_data.csv')
        data_tahunan = df.groupby('tahun')['jumlah_kasus'].sum().reset_index()
        
        # Initialize scaler
        scaler = MinMaxScaler()
        scaler.fit(data_tahunan[['tahun', 'jumlah_kasus']])
        
        return True
    except Exception as e:
        print(f"Error loading model and data: {str(e)}")
        return False

def predict_cases(year):
    try:
        # Normalisasi tahun input
        year_scaled = scaler.transform([[year, 0]])[0, 0].reshape(-1, 1)
        
        # Prediksi dengan model
        prediction_scaled = model.predict(year_scaled)
        
        # Balikkan skala hasil prediksi
        prediction = scaler.inverse_transform([[year_scaled[0, 0], prediction_scaled[0, 0]]])[0, 1]
        
        return int(prediction)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

@app.route('/')
def home():
    if model is None:
        success = load_model_and_data()
        if not success:
            return "Error loading model and data. Please try again later.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please refresh the page.'
        })
    
    try:
        # Ambil input tahun dari form
        year = int(request.form['year'])
        
        # Lakukan prediksi
        prediction = predict_cases(year)
        
        # Buat plot dengan style dark
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Plot 1: Data Aktual
        sns.scatterplot(data=data_tahunan, x='tahun', y='jumlah_kasus', color='#00ff88', label='Data Aktual', ax=ax1)
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Jumlah Kasus')
        ax1.set_title('Tren Kasus Pneumonia di Jawa Barat')
        ax1.set_facecolor('#2d2d2d')
        
        # Plot 2: Hasil Prediksi vs Data Aktual
        ax2.scatter(data_tahunan['tahun'], data_tahunan['jumlah_kasus'], color='#00ff88', label='Data Aktual')
        ax2.scatter(year, prediction, color='red', s=100, label='Prediksi ANN')
        
        # Atur range tahun untuk plot kedua
        min_year = min(data_tahunan['tahun'].min(), year)
        max_year = max(data_tahunan['tahun'].max(), year)
        ax2.set_xlim(min_year - 1, max_year + 1)
        
        # Tambahkan garis trend
        all_years = np.append(data_tahunan['tahun'].values, year)
        all_cases = np.append(data_tahunan['jumlah_kasus'].values, prediction)
        z = np.polyfit(all_years, all_cases, 1)
        p = np.poly1d(z)
        trend_x = np.linspace(min_year, max_year, 100)
        ax2.plot(trend_x, p(trend_x), "r--", alpha=0.8, label='Trend')
        
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Jumlah Kasus')
        ax2.set_title('Hasil Prediksi ANN vs Data Aktual')
        ax2.legend()
        ax2.set_facecolor('#2d2d2d')
        
        # Atur layout dan simpan ke memory
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='#1a1a1a')
        plt.close()
        buf.seek(0)
        
        # Convert plot ke base64
        plot_url = base64.b64encode(buf.getvalue()).decode()
        
        return jsonify({
            'status': 'success',
            'year': year,
            'prediction': prediction,
            'plot_url': f'data:image/png;base64,{plot_url}'
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
