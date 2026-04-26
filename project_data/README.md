# ⚒️ DataForge AI | Dataset Diagnosis & Auto-Fix System

**DataForge AI** is a premium, AI-driven platform designed to transform raw, messy datasets into high-fidelity, machine-learning-ready assets. Built with a "Data Scientist in a Box" philosophy, it automates the most time-consuming parts of the data workflow: **Diagnosis, Cleaning, and Model Selection.**

---

## 🚀 Key Features

### 🔍 **AI-Powered Dataset Diagnosis**
- **Health Score**: Instant 0–100% rating of your dataset's quality.
- **Deep Audit**: Automatic detection of missing values, hidden duplicates, constant columns, and statistical outliers.
- **Smart Insights**: Procedurally generated insights about data distribution and column relationships.

### ✨ **One-Click Auto-Cleaning**
- **Intelligent Imputation**: Fills missing values with statistically sound strategies.
- **Normalization**: Standardizes text and numeric formats.
- **Deduplication**: Deep scanning and removal of redundant rows and columns.
- **100% Health Guarantee**: Automatically forge a perfect dataset ready for production.

### 🧠 **Intelligent ML Induction**
- **Target Analysis**: Automatically detects if your goal is **Classification** or **Regression**.
- **Model induction**: Automatically chooses, trains, and tunes the best algorithm for your specific data.
- **Confidence Metrics**: Real-time evaluation with "Confidence Badges" and model health diagnostics.
- **Feature Importance**: Visualizes the strongest "drivers" behind your data’s patterns.

### 📱 **Mobile-First Luxury UI**
- **Laptop-Style Horizontal Flow**: A unique, swipeable dashboard that brings a desktop-rich experience to mobile devices.
- **Peek UX**: Smart mobile layout that "peeks" the next section, making navigation intuitive and obvious.
- **Neon Branding**: A high-end, futuristic aesthetic with interactive glowing elements and smooth micro-animations.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with advanced Custom CSS and JavaScript injection.
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) (Python) for ultra-fast data processing.
- **Database**: [MongoDB](https://www.mongodb.com/) for scalable dataset and model storage.
- **Data Science**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Scikit-Learn](https://scikit-learn.org/).
- **Visualization**: [Plotly Express](https://plotly.com/python/plotly-express/) for interactive, responsive charts.

---

## 🏁 Quick Start

### 1. **Prerequisites**
- Python 3.9+
- MongoDB (Local or Atlas)

### 2. **Installation**
Clone the repository and install dependencies:
```bash
git clone [repository-url]
cd [repository-folder]
pip install -r requirements.txt
```

### 3. **Environment Setup**
Create a `.env` file in the root:
```env
MONGODB_URI=your_mongodb_connection_string
BACKEND_URL=http://localhost:8000
```

### 4. **Launch the Engine**
**Start the Backend:**
```bash
python main.py
```

**Start the Frontend:**
```bash
streamlit run frontend/streamlit_app.py
```

---

## 🌐 Deployment

- **Frontend**: Optimized for [Streamlit Cloud](https://streamlit.io/cloud).
- **Backend**: Ready for [Render](https://render.com/) or Railway.
- **Database**: Recommended [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).

---

## 🛡️ License
Designed for maximum intelligence. Built by **Antigravity**. ⚒️✨
