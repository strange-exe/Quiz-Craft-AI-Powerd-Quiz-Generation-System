"""
QuizCraft AI — app.py  (College Capstone Project)
==================================================
✅ SQLite DB — users survive server restarts
✅ Bcrypt password hashing via Werkzeug
✅ Forgot password — shows reset link ON SCREEN (no email server needed)
✅ Real reset-password form with token validation
✅ Google/GitHub OAuth → "coming soon" page
✅ Quiz count per user tracked in DB
✅ All quiz routes protected with @login_required
"""

import os, random, json, time
from datetime import datetime, timedelta
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

import numpy as np
import faiss
import PyPDF2
from google import genai
from google.genai import types

# ── App & config ──────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]                     = os.environ.get("SECRET_KEY", "quizcraft-capstone-2025-secret")
app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///quizcraft.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"]     = timedelta(days=7)

db         = SQLAlchemy(app)
serialiser = URLSafeTimedSerializer(app.config["SECRET_KEY"])

# ── User model ────────────────────────────────────────
class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer,     primary_key=True)
    first_name    = db.Column(db.String(80),  nullable=False)
    last_name     = db.Column(db.String(80),  nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    quiz_count    = db.Column(db.Integer,     default=0)
    joined_at     = db.Column(db.DateTime,    default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

# ── Auth helper ───────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def current_user():
    return User.query.get(session["user_id"]) if "user_id" in session else None

# ── Gemini / RAG ──────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
client_ai        = genai.Client(api_key=GEMINI_API_KEY)
GENERATION_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL  = "gemini-embedding-001"
CHUNK_SIZE, CHUNK_OVERLAP, MAX_RETRIES = 250, 20, 5

def call_gemini_with_retry(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = (2 ** (attempt + 1)) + random.uniform(0, 1)
                print(f"[Quota] Wait {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise e
    return None

def extract_text(pdf_file, start_page, end_page):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text, total = "", len(reader.pages)
        for i in range(max(0, start_page - 1), min(total, end_page)):
            t = reader.pages[i].extract_text()
            if t: text += t + "\n"
        return text.strip()
    except Exception:
        return ""

def chunk_text(text):
    if not text: return []
    words = text.split()
    return [" ".join(words[i:i+CHUNK_SIZE])
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP)]

def build_index(chunks):
    if not chunks: return None
    res = call_gemini_with_retry(
        client_ai.models.embed_content, model=EMBEDDING_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
    if not res: return None
    emb = np.array([e.values for e in res.embeddings]).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb)
    return idx

def generate_mcqs_rag(chunks, vector_index, num_questions):
    if not chunks: return []
    ctx = ""
    if vector_index:
        anchor = random.choice(chunks)
        res = call_gemini_with_retry(
            client_ai.models.embed_content, model=EMBEDDING_MODEL,
            contents=anchor,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        if res:
            qv = np.array([res.embeddings[0].values]).astype("float32")
            _, ids = vector_index.search(qv, 2)
            ctx = "\n\n".join(chunks[i] for i in ids[0] if i < len(chunks))
    if not ctx:
        ctx = "\n\n".join(random.sample(chunks, min(len(chunks), 2)))
    ctx = ctx[:1500]
    schema = {"type":"ARRAY","items":{"type":"OBJECT","properties":{
        "type":{"type":"STRING"},"question":{"type":"STRING"},
        "options":{"type":"ARRAY","items":{"type":"STRING"}},"answer":{"type":"STRING"}},
        "required":["type","question","options","answer"]}}
    resp = call_gemini_with_retry(
        client_ai.models.generate_content, model=GENERATION_MODEL,
        contents=f"Create {num_questions} quiz questions based on this text:\n\n{ctx}",
        config=types.GenerateContentConfig(temperature=0.3,
            response_mime_type="application/json", response_schema=schema))
    return json.loads(resp.text) if resp else []

# ══════════════════════════════════════════════════════
#  PUBLIC ROUTES
# ══════════════════════════════════════════════════════
@app.route("/")
def home():
    return redirect(url_for("landing"))

@app.route("/landing")
def landing():
    return render_template("landing.html", scroll_to=None, contact_sent=False)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("landing.html", scroll_to="contact",
                           contact_sent=(request.method == "POST"))

@app.route("/terms")
def terms():
    return render_template("landing.html", scroll_to="footer", contact_sent=False)

@app.route("/privacy")
def privacy():
    return render_template("landing.html", scroll_to="footer", contact_sent=False)

# ══════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session: return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        first = request.form.get("first_name","").strip()
        last  = request.form.get("last_name", "").strip()
        email = request.form.get("email",     "").strip().lower()
        pwd   = request.form.get("password",  "")
        cpwd  = request.form.get("confirm_password", "")
        terms = request.form.get("terms")
        if not all([first, last, email, pwd, cpwd]):
            error = "Please fill in all fields."
        elif not terms:
            error = "You must agree to the Terms of Service."
        elif len(pwd) < 8:
            error = "Password must be at least 8 characters."
        elif pwd != cpwd:
            error = "Passwords do not match."
        elif User.query.filter_by(email=email).first():
            error = "An account with this email already exists."
        else:
            u = User(first_name=first, last_name=last, email=email)
            u.set_password(pwd)
            db.session.add(u)
            db.session.commit()
            return redirect(url_for("login") + "?success=Account+created!+Please+sign+in.")
    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session: return redirect(url_for("index"))
    error   = request.args.get("error")
    success = request.args.get("success")
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pwd   = request.form.get("password","")
        rem   = request.form.get("remember")
        user  = User.query.filter_by(email=email).first()
        if not user:
            error = "No account found with that email."
        elif not user.check_password(pwd):
            error = "Incorrect password. Please try again."
        else:
            session["user_id"]         = user.id
            session["user_first_name"] = user.first_name
            session["user_email"]      = user.email
            if rem: session.permanent  = True
            return redirect(url_for("index"))
    return render_template("login.html", error=error, success=success)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """
    No email server needed.
    We generate a real signed token and DISPLAY the link on screen.
    User copies it, opens it, and sets a new password — fully functional.
    """
    reset_link = None
    error      = None
    success    = None
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        user  = User.query.filter_by(email=email).first()
        if not user:
            error = "No account found with that email address."
        else:
            token      = serialiser.dumps(email, salt="pw-reset")
            reset_link = url_for("reset_password", token=token, _external=True)
            success    = "Reset link generated! Copy and open the link below:"
    return render_template("forgot_password.html",
                           reset_link=reset_link, error=error, success=success)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    error = None
    try:
        email = serialiser.loads(token, salt="pw-reset", max_age=3600)
    except SignatureExpired:
        return redirect(url_for("login") + "?error=Reset+link+expired.")
    except BadSignature:
        return redirect(url_for("login") + "?error=Invalid+reset+link.")
    user = User.query.filter_by(email=email).first()
    if not user: return redirect(url_for("login"))
    if request.method == "POST":
        pwd  = request.form.get("password","")
        cpwd = request.form.get("confirm_password","")
        if len(pwd) < 8:
            error = "Password must be at least 8 characters."
        elif pwd != cpwd:
            error = "Passwords do not match."
        else:
            user.set_password(pwd)
            db.session.commit()
            return redirect(url_for("login") + "?success=Password+updated!+Please+sign+in.")
    return render_template("reset_password.html", token=token, error=error, email=email)


@app.route("/auth/google")
@app.route("/auth/github")
def oauth_coming_soon():
    return render_template("oauth_soon.html")

# ══════════════════════════════════════════════════════
#  PROTECTED QUIZ ROUTES
# ══════════════════════════════════════════════════════
@app.route("/upload", methods=["GET", "POST"])
@login_required
def index():
    user = current_user()
    if request.method == "POST":
        pdf   = request.files.get("pdf")
        num   = min(int(request.form.get("num_questions", 5)), 10)
        start = int(request.form.get("start_page", 1))
        end   = int(request.form.get("end_page", 5))
        if pdf:
            text = extract_text(pdf, start, end)
            if not text:
                return render_template("index.html", error="No text found in those pages.",
                                       user_name=user.first_name, quiz_count=user.quiz_count)
            chunks = chunk_text(text)
            idx    = build_index(chunks)
            mcqs   = generate_mcqs_rag(chunks, idx, num)
            if not mcqs:
                return render_template("index.html",
                                       error="Quota exceeded. Wait 60s and retry.",
                                       user_name=user.first_name, quiz_count=user.quiz_count)
            user.quiz_count += 1
            db.session.commit()
            session["mcqs"] = mcqs
            return redirect(url_for("quiz"))
    return render_template("index.html", user_name=user.first_name, quiz_count=user.quiz_count)


@app.route("/quiz", methods=["GET", "POST"])
@login_required
def quiz():
    mcqs = session.get("mcqs", [])
    if not mcqs: return redirect(url_for("index"))
    if request.method == "POST":
        ans   = [request.form.get(f"q{i}","") for i in range(len(mcqs))]
        score = sum(1 for i,q in enumerate(mcqs)
                    if ans[i].strip().lower() == q["answer"].strip().lower())
        session["user_answers"] = ans
        session["score"]        = score
        return redirect(url_for("result"))
    return render_template("quiz.html", mcqs=mcqs)


@app.route("/result")
@login_required
def result():
    return render_template("result.html",
                           mcqs=session.get("mcqs",[]),
                           user_answers=session.get("user_answers",[]),
                           score=session.get("score",0))


@app.route("/restart")
@login_required
def restart():
    uid, uname, uemail = (session.get("user_id"),
                          session.get("user_first_name"),
                          session.get("user_email"))
    session.clear()
    session["user_id"]         = uid
    session["user_first_name"] = uname
    session["user_email"]      = uemail
    return redirect(url_for("index"))


# ══════════════════════════════════════════════════════
#  INIT DB + RUN
# ══════════════════════════════════════════════════════
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)