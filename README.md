# 🛒 Sales Forecasting for Retail

## 📌 Project Overview  
Retailers face a critical challenge:  
- **Understock** → Lost sales & unhappy customers  
- **Overstock** → Wasted money & storage costs  

This project builds **machine learning and deep learning models** to forecast retail sales and provide decision support for inventory management.  
It includes an **interactive Streamlit dashboard** where users can explore forecasts and stock risk levels.

---

## ⚙️ Features
✅ Multiple forecasting models:  
- **XGBoost Regressor**  
- **LSTM (Deep Learning)**  
- **ARIMA (Time Series)**  

✅ Streamlit dashboard with:  
- Store-wise forecast visualization  
- Comparison of different models  
- **Stock Risk Alerts** (Understock ⚠️ & Overstock 💰)  
- Downloadable forecast CSV  

✅ Jupyter Notebook for model training & evaluation  

✅ Clear metrics & comparison visualizations  

---
```
## 📂 Project Structure
Sales-Forecasting-Retail/
│── data/                         # Dataset (sales.csv)
│── notebooks/                    # Jupyter notebooks
│ └── sales_forecasting.ipynb
│── outputs/                      # Model metrics & plots
│── dashboard/                    # Streamlit dashboard app
│ └── app.py
│── requirements.txt              # Dependencies
│── README.md                     # Project documentation
```

---

## 📊 Model Comparison  
- Models are evaluated using **RMSE, MAE, R²**  
- Best performing models are highlighted in the dashboard  

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sarandasari11/Sales-Forecasting-Retail.git
cd Sales-Forecasting-Retail

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
jupyter notebook notebooks/sales_forecasting.ipynb

cd dashboard
streamlit run app.py

```
---

## 📈 Results & Insights
- XGBoost captured short-term sales patterns effectively.  
- LSTM learned seasonal trends better over longer horizons.  
- ARIMA served as a strong statistical baseline.  
- Stock risk alerts help retailers balance between **understocking** and **overstocking**.  

---

## 🔮 Future Improvements
- Add external factors (holidays, promotions, weather).  
- Experiment with **Prophet & Transformer models**.  
- Deploy with **Docker + Cloud** for production use.  
- Extend to multi-store and multi-product forecasting.  

---

## 👨‍💻 Author
**Saran Dasari**  
📌 GitHub: [sarandasari11](https://github.com/sarandasari11)  
📧 Contact: dasarisaran2005@gmail.com

---
✨ Thank you for checking out this project!  
If you find it useful, please ⭐ the repo on GitHub.
