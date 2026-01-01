# 🔍 Fake News Detection Web App

A **multimodal AI web application** to detect fake and real news using **text and image analysis**, built with **Flask**, **PyTorch**, **Transformers (DistilBERT)**, **MobileNetV2**, and **SQLite**.
Includes **live news analysis**, **user authentication**, **admin panel**, **history tracking**, **charts**, and **auto dark mode**.

---

## 🛠 Features

### 1. User Management

* **User registration & login**
* Role-based access: `Admin` vs `User`
* Admin can view all users
* Logout functionality

### 2. News Detection

* Manual news input: text + image
* Automatic live news fetching and prediction
* Probabilities displayed with confidence bars
* Fact-checking using **entity extraction (spaCy)** and NewsAPI
* Country-specific news (`US`, `Pakistan`, `India`)

### 3. History & Statistics

* Save user prediction history in **SQLite**
* View history table with timestamps
* **Charts**: Fake vs Real news (overall and by country)
* Dashboard showing country-wise statistics

### 4. UI/UX

* **Dark mode** with animated toggle
* Responsive design using **custom CSS**
* Auto-changing background images every 20 seconds
* Live news cards showing prediction automatically
* Loading spinner for predictions

---

## 📁 Project Structure

```
fake-news-app/
│
├── app.py                 # Flask application
├── fakeddit_multimodal_model_full.pth  # PyTorch model
├── database.db            # SQLite database
├── requirements.txt
├── render.yaml            # Render deployment config
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── admin.html
│   └── history.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── uploads/               # Temporary image uploads
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-app.git
cd fake-news-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Important:** Install the spaCy model directly:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
```

### 3. Set your NewsAPI key

```bash
export NEWS_API_KEY="YOUR_NEWSAPI_KEY"   # Linux/Mac
set NEWS_API_KEY="YOUR_NEWSAPI_KEY"      # Windows
```

### 4. Run Flask app locally

```bash
python app.py
```

Access via: `http://127.0.0.1:5001/`

---

## 🚀 Deployment on Render

1. Push your code to **GitHub**.
2. Go to [Render](https://render.com) → **New Web Service** → Connect repo.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
gunicorn app:app
```

5. Add **Environment Variable**:

```
Key: NEWS_API_KEY
Value: "bcf244511c0e4faeb96ff55167517c9f"
```

6. Deploy. Render URL will be generated automatically.

---

## 🧠 How It Works

1. **Text & Image Input** → Processed by **DistilBERT** + **MobileNetV2**
2. **Entity Extraction** using **spaCy** → Fact-check via NewsAPI
3. **Fusion Layer** combines text & image features
4. **Softmax output** → Fake or Real prediction
5. Results saved in **SQLite history table**

---

## 🌍 Live News Auto Detection

* Automatically fetches top 5 headlines from **NewsAPI**.
* Predicts fake/real for each headline with confidence.
* Country filter available (`US`, `PK`, `IN`).
* Updates every 5 minutes.

---

## 🎨 UI Features

* Dark/Light mode with animated toggle
* Responsive and mobile-friendly
* Live prediction results
* Auto-changing background images every 20 seconds
* Charts for visualization

---

## 📌 Notes

* SQLite database resets on redeploy (Render ephemeral filesystem)
* Model file `fakeddit_multimodal_model_full.pth` should be included in project root
* Live News requires **NewsAPI Key**: [https://newsapi.org/](https://newsapi.org/)
* Predictions are **for educational purposes only**

---

## 👨‍💻 Tech Stack

* Python 3.10+
* Flask
* PyTorch
* Transformers (HuggingFace)
* MobileNetV2
* spaCy
* SQLite
* HTML/CSS/JS (vanilla)
* Gunicorn (for production)
* Render (for deployment)

---

## 📄 License

MIT License
