# 📊 Demand Forecasting Web App using Prophet, ARIMA & Holt-Winters  

A **Time Series Demand Forecasting Application** built using **Prophet**, **ARIMA**, and **Holt-Winters Exponential Smoothing**.  
This project aims to analyze and predict future product demand using various time series models and visualize the results interactively through a **Streamlit web interface**.  

🔗 **Live App:** [https://demandforcatingtsa.streamlit.app/](https://demandforcatingtsa.streamlit.app/)

---

## 🚀 Features  

- 📁 Upload or use sample demand data (CSV format)  
- 📈 Forecast future demand using **Prophet**, **ARIMA**, and **Holt-Winters** models  
- 📊 Visualize model predictions and compare performance  
- 🧮 Automatically detects trend, seasonality, and volatility  
- ⚡ Simple and clean **Streamlit interface** for real-time forecasting  

---

## 🧠 Tech Stack  

| Component | Technology Used |
|------------|----------------|
| **Frontend/UI** | Streamlit |
| **Time Series Models** | Prophet, ARIMA, Holt-Winters |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Development Environment** | VS Code |
| **Language** | Python |

---

## 🗂️ Project Structure  

📦 tsa_demand_forecasting/
├── app.py # Main Streamlit app
├── demand_forcasting.ipynb # Jupyter Notebook for model analysis
├── temp-plot/ # Temporary folder for saved plots
├── dataset/ # Contains Tesla or other datasets
│ └── tesla.csv # Example dataset
├── requirements.txt # All dependencies
└── README.md # Project documentation


---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/tsa_demand_forecasting.git
cd tsa_demand_forecasting
```

### 2️⃣ Create a Virtual Environment
``` bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux
```

### 3️⃣ Install Dependencies
``` bash
pip install -r requirements.txt
```

### 4️⃣ Run the App Locally
``` bash
streamlit run app.py
```

## 📈 Models Used
### 🧩 1. ARIMA (AutoRegressive Integrated Moving Average)

  Used for univariate time series forecasting by capturing autocorrelations in the data.

#### 🧮 2. Holt-Winters Exponential Smoothing

  Captures trend and seasonality in time series data through level, trend, and seasonal components.

### 🔮 3. Prophet (by Meta/Facebook)

  A robust model for forecasting with automatic handling of trend shifts, holidays, and seasonality.

## 📊 Example Outputs

  Forecasted vs Actual Demand

  Interactive Time Series Graphs

  Model Performance Comparison (ARIMA vs Holt-Winters vs Prophet)

  Volatility and Trend Analysis

## 💡 Use Cases

  🏪 Retail & E-commerce Demand Prediction

  🏭 Manufacturing Production Planning
  
  🚚 Supply Chain Optimization

  💰 Stock or Energy Demand Forecasting

## 🌐 Deployment

  The project is deployed live on Streamlit Cloud and accessible at:
  👉 https://demandforcatingtsa.streamlit.app/

## 👨‍💻 Author

Shreyash Jain
📧 Jainshreyash89@gmail.com
💻 Passionate about AI, Machine Learning, and Cloud-based Applications

## 🏁 Future Enhancements

  Add model accuracy comparison metrics (MAE, RMSE, MAPE)

  Implement automated data preprocessing and missing value handling

  Support multivariate time series inputs

  Enable saving and exporting of forecasts
