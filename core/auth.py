"""
Authentication + audit log for MedPredict (SQLite, standard library only).

Roles
-----
admin   – seeded on first start; manages doctors and can see all prediction logs
doctor  – self-registers on the Register page; can run predictions and see own history

Security notes
--------------
* Passwords are stored as PBKDF2-HMAC-SHA256 hashes with a per-user random salt.
* 5 wrong passwords lock the account for 10 minutes.
* Change the default admin password after first login (or set ADMIN_EMAIL / ADMIN_PASSWORD
  as environment variables / Streamlit secrets *before* the first run).
"""
import contextlib
import datetime as dt
import hashlib
import hmac
import os
import re
import secrets
import sqlite3

DB_PATH = os.environ.get("MEDPREDICT_DB", os.path.join("data", "medpredict.db"))
ITERATIONS = 200_000
MAX_FAILS = 5
LOCK_MINUTES = 10
DEFAULT_ADMIN_EMAIL = "admin@medpredict.com"
DEFAULT_ADMIN_PASSWORD = "Admin@123"

SPECIALIZATIONS = [
    "General Physician", "Pulmonologist", "Oncologist", "Cardiologist", "Endocrinologist",
    "Neurologist", "Hepatologist / Gastroenterologist", "Nephrologist", "Radiologist",
    "Resident / Intern", "Other",
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _now() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


@contextlib.contextmanager
def _db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex()


def init_db(admin_email: str | None = None, admin_password: str | None = None) -> None:
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','doctor')),
            specialization TEXT, license_no TEXT, hospital TEXT,
            phone TEXT, listed INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','disabled')),
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            default_password INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            disease TEXT NOT NULL, patient_name TEXT,
            result TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)
        # migrate databases created by the earlier version
        cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
        if "phone" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        if "listed" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN listed INTEGER NOT NULL DEFAULT 0")
        has_admin = c.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone()
        if not has_admin:
            email = (admin_email or DEFAULT_ADMIN_EMAIL).strip().lower()
            pw = admin_password or DEFAULT_ADMIN_PASSWORD
            salt = secrets.token_bytes(16)
            c.execute(
                "INSERT INTO users (full_name,email,password_hash,salt,role,default_password,created_at) "
                "VALUES (?,?,?,?, 'admin', ?, ?)",
                ("Administrator", email, _hash(pw, salt), salt.hex(),
                 1 if (pw == DEFAULT_ADMIN_PASSWORD) else 0, _now()))


# ───────────────────────────── validation ─────────────────────────────
def normalize_phone(raw: str) -> str | None:
    """Accepts +91 98765 43210 / 98765-43210 etc. Returns cleaned number or None if invalid."""
    raw = (raw or "").strip()
    digits = re.sub(r"\D", "", raw)
    if not re.fullmatch(r"\+?[\d\s\-()]+", raw) or not (10 <= len(digits) <= 15):
        return None
    return ("+" if raw.startswith("+") else "") + digits


def validate_password(pw: str) -> str | None:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return "Password must contain at least one letter and one number."
    return None


def register_doctor(full_name, email, password, confirm, specialization, license_no, hospital="",
                    phone="", consent=False):
    """Returns (ok, message)."""
    full_name, email = (full_name or "").strip(), (email or "").strip().lower()
    license_no, hospital = (license_no or "").strip(), (hospital or "").strip()
    if not full_name or not email or not password or not license_no or not (phone or "").strip():
        return False, "Please fill in name, email, mobile number, medical registration number and password."
    phone_clean = normalize_phone(phone)
    if not phone_clean:
        return False, "Please enter a valid mobile number (10–15 digits, optional +country code)."
    if not consent:
        return False, "Please agree to be listed in the public doctor directory to register."
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if password != confirm:
        return False, "Passwords do not match."
    err = validate_password(password)
    if err:
        return False, err
    salt = secrets.token_bytes(16)
    try:
        with _db() as c:
            c.execute(
                "INSERT INTO users (full_name,email,password_hash,salt,role,specialization,license_no,hospital,phone,listed,created_at) "
                "VALUES (?,?,?,?, 'doctor', ?,?,?,?,1,?)",
                (full_name, email, _hash(password, salt), salt.hex(), specialization, license_no, hospital,
                 phone_clean, _now()))
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    return True, "Account created. You can sign in now."


def authenticate(email, password, role):
    """Returns (user_dict | None, error_message | None)."""
    email = (email or "").strip().lower()
    generic = "Incorrect email or password."
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row is None:
            return None, generic
        if row["locked_until"] and row["locked_until"] > _now():
            return None, f"Too many failed attempts. Try again after {LOCK_MINUTES} minutes."
        ok = hmac.compare_digest(_hash(password or "", bytes.fromhex(row["salt"])), row["password_hash"])
        if not ok:
            fails = row["failed_attempts"] + 1
            lock = None
            if fails >= MAX_FAILS:
                lock = (dt.datetime.utcnow() + dt.timedelta(minutes=LOCK_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
                fails = 0
            c.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?", (fails, lock, row["id"]))
            return None, generic
        if row["role"] != role:
            return None, f"This account is not registered as {'an admin' if role == 'admin' else 'a doctor'}."
        if row["status"] != "active":
            return None, "This account has been disabled. Please contact the administrator."
        c.execute("UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE id=?",
                  (_now(), row["id"]))
        user = {k: row[k] for k in ("id", "full_name", "email", "role", "specialization", "default_password")}
        return user, None


def change_password(user_id, old_pw, new_pw):
    err = validate_password(new_pw)
    if err:
        return False, err
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None or not hmac.compare_digest(_hash(old_pw or "", bytes.fromhex(row["salt"])), row["password_hash"]):
            return False, "Current password is incorrect."
        salt = secrets.token_bytes(16)
        c.execute("UPDATE users SET password_hash=?, salt=?, default_password=0 WHERE id=?",
                  (_hash(new_pw, salt), salt.hex(), user_id))
    return True, "Password updated."


# ───────────────────────────── admin helpers ─────────────────────────────
def list_doctors():
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, full_name, email, phone, listed, specialization, license_no, hospital, status, created_at, last_login "
            "FROM users WHERE role='doctor' ORDER BY created_at DESC")]


def set_status(user_id: int, status: str) -> None:
    with _db() as c:
        c.execute("UPDATE users SET status=? WHERE id=? AND role='doctor'", (status, user_id))


def delete_doctor(user_id: int) -> None:
    with _db() as c:
        c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE id=? AND role='doctor'", (user_id,))


# ───────────────────────────── public doctor directory ─────────────────────────────
def list_directory():
    """Active doctors who opted in to the public directory."""
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT full_name, specialization, hospital, phone, email FROM users "
            "WHERE role='doctor' AND status='active' AND listed=1 ORDER BY full_name COLLATE NOCASE")]


def get_contact(user_id):
    with _db() as c:
        r = c.execute("SELECT phone, hospital, listed FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(r) if r else {}


def update_contact(user_id, phone, hospital, listed):
    phone_clean = normalize_phone(phone) if (phone or "").strip() else None
    if listed and not phone_clean:
        return False, "A valid mobile number is required to be listed in the directory."
    if (phone or "").strip() and not phone_clean:
        return False, "Please enter a valid mobile number (10–15 digits, optional +country code)."
    with _db() as c:
        c.execute("UPDATE users SET phone=?, hospital=?, listed=? WHERE id=? AND role='doctor'",
                  (phone_clean, (hospital or "").strip(), 1 if listed else 0, user_id))
    return True, "Contact details updated."


# ───────────────────────────── prediction log ─────────────────────────────
def log_prediction(user_id, disease, patient_name, result):
    with _db() as c:
        c.execute("INSERT INTO predictions (user_id,disease,patient_name,result,created_at) VALUES (?,?,?,?,?)",
                  (user_id, disease, (patient_name or "").strip() or None, result, _now()))


def get_predictions(user_id=None, limit=500):
    q = ("SELECT p.id, p.created_at, p.disease, p.patient_name, p.result, u.full_name AS doctor "
         "FROM predictions p JOIN users u ON u.id=p.user_id ")
    args = ()
    if user_id is not None:
        q += "WHERE p.user_id=? "
        args = (user_id,)
    q += "ORDER BY p.id DESC LIMIT ?"
    with _db() as c:
        return [dict(r) for r in c.execute(q, args + (limit,))]


def stats():
    with _db() as c:
        one = lambda sql: c.execute(sql).fetchone()[0]
        return {
            "doctors": one("SELECT COUNT(*) FROM users WHERE role='doctor'"),
            "active": one("SELECT COUNT(*) FROM users WHERE role='doctor' AND status='active'"),
            "predictions": one("SELECT COUNT(*) FROM predictions"),
            "positive": one("SELECT COUNT(*) FROM predictions WHERE result='Positive'"),
        }


def by_disease():
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT disease, COUNT(*) AS n FROM predictions GROUP BY disease ORDER BY n DESC")]
