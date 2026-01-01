from random import random
import os, sqlite3, torch
import time
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from transformers import AutoTokenizer, DistilBertModel
from PIL import Image
import requests
import spacy

# ==============================
# Flask Config
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# Database Init
# ==============================
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    # Create users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )""")

    # Create history table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        text TEXT,
        prediction TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit()
    conn.close()

init_db()

# ==============================
# Model Setup
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(768, 256)
        self.image_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.image_model.classifier = nn.Identity()
        self.image_fc = nn.Linear(1280, 256)
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2))
    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_model(input_ids, attention_mask).last_hidden_state[:,0]
        text_feat = self.text_fc(text_out)
        img_feat = self.image_fc(self.image_model(image))
        fused = torch.cat([text_feat, img_feat], dim=1)
        return self.classifier(fused)

model = MultiModel().to(device)
model.load_state_dict(torch.load("fakeddit_multimodal_model_full.pth", map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# ==============================
# spaCy Fact-check Setup
# ==============================
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extract DATE, GPE, ORG, PERSON entities from text."""
    doc = nlp(text)
    entities = {"DATE":[], "GPE":[], "ORG":[], "PERSON":[]}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

NEWS_API_KEY = os.getenv("bcf244511c0e4faeb96ff55167517c9f") # Replace with your NewsAPI key

def fact_check(title, entities, max_articles=5):
    """Check extracted entities against live news using NewsAPI."""
    if not entities: return True  # No entities to check, assume valid
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": title,
        "apiKey": NEWS_API_KEY,
        "pageSize": max_articles
    }
    try:
        resp = requests.get(url, params=params).json()
        articles = resp.get("articles", [])
        if not articles: return False
        for key in entities:
            for ent in entities[key]:
                ent_lower = ent.lower()
                found = any(ent_lower in (a["title"].lower() + " " + str(a.get("description","")).lower()) for a in articles)
                if not found: return False
        return True
    except:
        return True

# ==============================
# Auth Routes
# ==============================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method=="POST":
        u = request.form["username"]
        p = request.form["password"]
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE username=? AND password=?", (u,p))
        row = cur.fetchone()
        conn.close()
        if row:
            session["user"] = u
            session["role"] = row[0]
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        # Check if there are any users
        cur.execute("SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]

        # First user -> admin, else -> user
        role = "admin" if user_count == 0 else "user"

        try:
            cur.execute("INSERT INTO users(username,password,role) VALUES(?,?,?)", 
                        (username, password, role))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="User already exists")

        conn.close()
        return redirect("/")

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ==============================
# Pages
# ==============================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")

@app.route("/detect")
def detect():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html")

@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/")

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT text,prediction,confidence,country,created_at
        FROM history
        WHERE username=?
        ORDER BY created_at DESC
    """, (session["user"],))
    rows = cur.fetchall()
    conn.close()

    return render_template("history.html", rows=rows)


@app.route("/admin")
def admin():
    if session.get("role")!="admin":
        return "Forbidden",403
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT username,role FROM users")
    users = cur.fetchall()
    conn.close()
    return render_template("admin.html", users=users)

# ==============================
# Prediction API with Fact-check
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    image = request.files.get("image")
    country = request.form.get("country", "us")

    if not text or not image:
        return jsonify({"status":"error","message":"Missing input"})

    filename = secure_filename(image.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")

    with torch.no_grad():
        out = model(enc["input_ids"].to(device),
                    enc["attention_mask"].to(device), img)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs).item()

    result = {
        "status":"success",
        "prediction":"Real News" if pred==1 else "Fake News",
        "confidence":round(probs[0][pred].item()*100,2),
        "fake_probability":round(probs[0][0].item()*100,2),
        "real_probability":round(probs[0][1].item()*100,2)
    }

    # Save with country
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history(username,text,prediction,confidence,country)
        VALUES(?,?,?,?,?)
    """, (
        session.get("user","guest"),
        text,
        result["prediction"],
        result["confidence"],
        country
    ))
    conn.commit()
    conn.close()

    os.remove(path)
    return jsonify(result)


# ==============================
# Stats & Live News APIs
# ==============================
@app.route("/country-stats")
def country_stats():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT country,
        SUM(CASE WHEN prediction='Fake News' THEN 1 ELSE 0 END) AS fake,
        SUM(CASE WHEN prediction='Real News' THEN 1 ELSE 0 END) AS real
        FROM history
        GROUP BY country
    """)

    rows = cur.fetchall()
    conn.close()

    data = {}
    for r in rows:
        data[r[0]] = {"fake": r[1], "real": r[2]}

    return jsonify(data)


import random, time, requests
from flask import jsonify, request

@app.route("/live-news")
def live_news():
    country = request.args.get("country", "us")  # default US

    # Allowed countries
    allowed = {"us": "us", "pk": "pk", "in": "in"}
    country = allowed.get(country, "us")

    max_articles = 100

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": country,
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
        "_": int(time.time())  # cache buster
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        articles = data.get("articles", [])

        if not articles:
            return jsonify({"status": "error", "message": "No news found"})

        random.shuffle(articles)  # ✅ correct shuffle

        news = []
        for item in articles[:5]:
            news.append({
                "title": item.get("title") or "",
                "description": item.get("description") or "",
                "image": item.get("urlToImage") or "/static/images/default.jpg"
            })

        return jsonify({"status": "success", "articles": news})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    print("🚀 Fake News Detection App with Fact-check Running")
    app.run(host="0.0.0.0", port=5001, debug=True)