# 📊 Flipkart Sales Optimization using K-Means & Linear Regression

## 🚀 Project Overview

This project analyzes Flipkart sales data to optimize business performance using customer segmentation and revenue forecasting. By applying machine learning techniques, we aim to improve pricing strategy, demand prediction, and profit maximization.

The interactive Streamlit dashboard provides visualizations, predictive modeling, and business intelligence for data-driven decision-making in e-commerce sales optimization.

## 🌐 Live Demo

Streamlit App: [https://flipkartsalesperformanceoptimization-d3ivpuwumk8myxjmrcdbfr.streamlit.app/](https://flipkartsalesperformanceoptimization-d3ivpuwumk8myxjmrcdbfr.streamlit.app/)

---

# 📌 Business Problem Statement

Flipkart operates in a highly competitive e-commerce market where managing demand, pricing strategies, and customer targeting is critical. 

The key challenges include:

- Identifying high-value and low-value customers
- Forecasting revenue accurately
- Optimizing discount and pricing strategies
- Improving profit margins
- Reducing financial risk due to demand fluctuations

This project aims to solve these challenges using data-driven insights derived from clustering and predictive modeling.

---

# 📚 Economic Concepts Applied

This project integrates economic and financial principles with data science techniques:

- **Demand-Supply Analysis** – Understanding how quantity impacts revenue.
- **Price Elasticity** – Measuring how discounts affect sales performance.
- **Revenue Maximization** – Identifying high-performing customer segments.
- **Profit Maximization** – Analyzing cost vs revenue to improve margins.
- **Customer Lifetime Value (CLV)** – Segmenting customers using RFM.
- **Risk Analysis** – Evaluating prediction errors using RMSE.

---

# 🤖 AI Techniques Used

### 1️⃣ K-Means Clustering
- Applied on RFM (Recency, Frequency, Monetary) metrics.
- Segments customers into different clusters such as:
  - Loyal Customers
  - High-Value Customers
  - At-Risk Customers
  - Occasional Buyers

### 2️⃣ Linear Regression
- Used to forecast revenue.
- Independent Variables:
  - Quantity Sold
  - Discount
  - Month (Seasonality)
- Model Evaluation:
  - R² Score
  - RMSE (Root Mean Squared Error)

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
cd /Users/riddhizunjarrao/Desktop/Business\ Mini\ Project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Kaggle Credentials (Optional)
If you want to download the dataset directly from Kaggle:
1. Download `kaggle.json` from your Kaggle account settings
2. Place it in `~/.kaggle/kaggle.json`
3. Run `chmod 600 ~/.kaggle/kaggle.json` (macOS/Linux)

### Step 4: Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

# 📂 Dataset Link

Flipkart Sales Dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/iyumrahul/flipkartsalesdataset

---

# 📁 Project Structure

```
Business Mini Project/
├── app.py                          # Main Streamlit dashboard
├── Flipkart_Sales_Dataset.ipynb   # Jupyter notebook with analysis
├── Sales.csv                       # Main sales dataset
├── sales_preprocessed.pkl          # Cached preprocessed dataset (auto-generated)
├── overview_snapshot.pkl           # Cached overview snapshot (auto-generated)
├── products.csv                    # Products information
├── requirements.txt                # Python dependencies
├── kaggle.json                     # Kaggle API credentials
├── Readme.md                       # Project documentation
└── .streamlit/                     # Streamlit local settings
```

---

# 📊 Project Workflow

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. RFM Customer Segmentation using K-Means  
4. Revenue Prediction using Linear Regression  
5. Profit Analysis & Business Interpretation  
6. Deployment using Streamlit  

---

# 📈 Dashboard Features

The interactive Streamlit dashboard includes four main sections:

### 1️⃣ **Overview**
- Fast-loading KPI dashboard
- Monthly revenue/quantity trends
- Distribution and top-city views
- Lightweight table preview (100/200 rows)

### 2️⃣ **Exploratory Data Analysis (EDA)**
- **Monthly Revenue Trend**: Visualizes seasonal patterns in revenue
- **Top 10 Revenue Cities**: Bar chart showing highest-performing regions
- **Discount vs Revenue**: Scatter plot analyzing price elasticity
- **Correlation Heatmap**: Shows relationships between key variables
  - Quantity, Selling Price, Discount, Revenue

### 3️⃣ **Customer Segmentation (Clustering)**
- Interactive K-Means clustering algorithm
- Adjustable number of clusters (2-10)
- Customer aggregation by RFM metrics:
  - Total Quantity Purchased
  - Total Revenue Generated
  - Total Discounts Received
  - Number of Orders
- **Scatter/3D Visualization**: Sampled for browser-safe rendering
- **Cluster Statistics**: Mean values for each segment
- **Business Insight**: Identifies high-value, medium, and low-value customers

### 4️⃣ **Revenue Prediction**
- **Machine Learning Model**: Linear Regression with real-time predictions
- **Input Features**:
  - Quantity (number of items)
  - Discount amount
  - Month (for seasonality)
- **Model Performance Metrics**:
  - R² Score (model accuracy)
  - RMSE (prediction error)
- **Profit Analysis**: Monthly profit trends with visualization

---

# 📊 Key Insights & Findings

- **High-frequency customers** contribute significantly to revenue
- **Excessive discounts** reduce profit margins (inverse relationship)
- **Revenue is strongly influenced** by quantity sold
- **Certain clusters represent** premium loyal customers following Pareto Principle (80/20 rule)
- **Forecasting model** helps reduce revenue prediction risk
- **Seasonal patterns** indicate demand fluctuations throughout the year
- **City-level analysis** shows concentration of revenue in specific regions

---

# 🤖 Machine Learning Algorithms

### 1. **K-Means Clustering**
**Purpose**: Segment customers into distinct groups for targeted marketing

**How it works**:
- Groups similar customers based on purchasing behavior
- Uses features: Quantity, Revenue, Discount, Total Orders
- Data is standardized before clustering (StandardScaler)
- Elbow method used to determine optimal clusters

**Output**:
- Customer clusters labeled 0, 1, 2, etc.
- Cluster profiles with average metrics
- Actionable insights for pricing and loyalty programs

### 2. **Linear Regression**
**Purpose**: Predict revenue based on quantity, discount, and seasonality

**Formula**: 
```
Revenue = β₀ + β₁(Quantity) + β₂(Discount) + β₃(Month) + ε
```

**Model Evaluation**:
- **R² Score**: Measures how well the model explains revenue variance (0-1, higher is better)
- **RMSE**: Root Mean Squared Error - average prediction error in rupees
- **Train-Test Split**: 80% training, 20% testing data

**Interpretation**:
- Coefficients show the impact of each variable on revenue
- A high R² indicates strong predictive power
- Low RMSE indicates accurate predictions

---

# 🌐 Deployment & Usage

### Live Deployment (Streamlit Cloud)

- App URL: [https://flipkartsalesperformanceoptimization-d3ivpuwumk8myxjmrcdbfr.streamlit.app/](https://flipkartsalesperformanceoptimization-d3ivpuwumk8myxjmrcdbfr.streamlit.app/)

### Local Deployment with Streamlit

The project is deployed using **Streamlit**, a Python library for building interactive data applications.

**Features of the Streamlit Dashboard**:
- 🎯 **No backend required** - pure Python
- 📊 **Interactive visualizations** - real-time plots and charts
- ⚡ **Fast reloads** - automatic updates on code changes
- 📱 **Responsive design** - works on desktop and mobile
- 🎨 **Custom caching** - @st.cache_data for performance optimization

### How to Use the Dashboard

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate Sections** using the sidebar:
   - Select "Overview", "EDA", "Clustering", or "Prediction"

3. **Interact with Features**:
   - Adjust cluster count slider on Clustering page
   - Input quantity, discount, and month for predictions
   - Hover over charts for detailed information

4. **Export & Share**:
   - Use browser's screenshot tool for reports
   - Data tables can be downloaded via Streamlit's built-in features

---

# 🎯 Business Value

By combining K-Means clustering and Linear Regression, this project provides:

- **📈 Revenue Growth**: Targeted customer management increases repeat purchases
- **💰 Profit Optimization**: Data-driven pricing reduces margin erosion
- **🎯 Precision Marketing**: Segment-specific campaigns improve conversion rates
- **📊 Risk Reduction**: Accurate forecasting minimizes demand forecast errors
- **⏱️ Time Efficiency**: Automated analysis replaces manual reporting
- **🔍 Competitive Advantage**: Data-driven insights inform strategic decisions

---

# 👩‍💻 Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | Machine learning algorithms |
| **Streamlit** | Interactive web-based dashboard |
| **Jupyter** | Exploratory analysis and documentation |

---

# 📚 File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard application |
| `Flipkart_Sales_Dataset.ipynb` | Jupyter notebook with detailed analysis |
| `Sales.csv` | Main sales dataset (large source file) |
| `sales_preprocessed.pkl` | Preprocessed cache generated on first run |
| `overview_snapshot.pkl` | Compact cache used by Overview page |
| `products.csv` | Product information |
| `requirements.txt` | Python package dependencies |

---

# 📌 Configuration

### Data Loading
- Full dataset is loaded lazily (only when required by EDA/Clustering/Prediction pages)
- Overview uses a compact cached snapshot for faster startup
- Data is cached using `@st.cache_data` and `pickle` cache files
- Model training is lazy and starts only when opening the Prediction page

### Model Parameters
- **K-Means**: 3 clusters by default (adjustable in UI)
- **Linear Regression**: 80-20 train-test split
- **Scaling**: StandardScaler for feature normalization

---

# 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| *Module not found error* | Run `pip install -r requirements.txt` |
| *Streamlit not found* | Install with `pip install streamlit` |
| *CSV file not found* | Ensure `Sales.csv` is in the project directory |
| *MessageSizeError (data exceeds 200 MB)* | Reduce data sent to browser: use sampled charts, avoid showing full dataframes, keep filters bounded (e.g., top cities only) |
| *Slow first load* | First run builds `sales_preprocessed.pkl` and `overview_snapshot.pkl`; subsequent runs are much faster |
| *Port 8501 already in use* | Run `streamlit run app.py --logger.level=debug --client.serverAddress=localhost --server.port=8502` |

---

# 📌 Future Improvements

- 🤖 Use Random Forest or XGBoost for better prediction accuracy
- 📈 Implement ARIMA/Prophet for time series forecasting
- 💎 Add Customer Lifetime Value (CLV) prediction
- 🔄 Integrate real-time data pipeline
- 🔐 Add user authentication and role-based access
- 📧 Automated email reports based on clustering insights
- 🎨 Enhanced UI/UX with advanced Streamlit features
- 📊 Export predictions to CSV/Excel

---

# 💡 How to Extend This Project

### Add New Features:
1. Create new functions in `data_utils.py`
2. Add new pages in `app.py` using `st.radio()` options
3. Create corresponding HTML templates if needed

### Integrate External Data:
1. Add data loading functions to `data_utils.py`
2. Merge with existing Sales.csv using pandas
3. Update caching and model training

### Deploy Online:
- **Streamlit Cloud**: Free deployment at https://streamlit.io/cloud
- **Heroku**: Traditional Python app hosting
- **AWS/Azure**: Enterprise deployment options

---

# 📞 Contact & Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for detailed analysis
3. Examine function docstrings in the code

---

# 📄 License

This project uses the Flipkart Sales Dataset from Kaggle.
Ensure compliance with dataset and library licenses.
