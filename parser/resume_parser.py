import re
import spacy

nlp = spacy.load("en_core_web_sm")


# ---------------- EMAIL ----------------
def extract_email(text):
    m = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
    return m.group() if m else None


# ---------------- PHONE ----------------
def extract_phone(text):
    m = re.search(r'(\+?\d{1,3}[\s-]?)?\d{10}', text)
    return m.group() if m else None


# ---------------- NAME (NLP + HEADER FALLBACK) ----------------
def extract_name(text):

    doc = nlp(text[:1000])

    STOP_WORDS = [
        "github", "linkedin", "email", "phone",
        "summary", "experience", "skills"
    ]

    # -------- NLP Detection --------
    for ent in doc.ents:
        if ent.label_ == "PERSON":

            name = ent.text.replace("\n", " ").strip()

            parts = name.split()

            cleaned = []

            for p in parts:
                if p.lower() in STOP_WORDS:
                    break
                cleaned.append(p)

            name = " ".join(cleaned)

            # Limit max words (avoid extra tokens)
            name = " ".join(name.split()[:4])

            return name

    # -------- Fallback --------
    lines = text.split("\n")

    for l in lines[:10]:
        l = l.strip()
        if 2 <= len(l.split()) <= 4:
            return l

    return None



# ---------------- SKILLS (EXPANDED) ----------------
SKILL_SET = {
# ---------- Programming ----------
"python","java","c","c++","c#","go","rust","scala","kotlin","swift",
"javascript","typescript","php","ruby","matlab","r","perl","dart",

# ---------- Web ----------
"html","css","sass","less","react","angular","vue","nextjs","nuxt",
"node","express","django","flask","fastapi","spring","spring boot",

# ---------- Data Science ----------
"machine learning","deep learning","nlp","computer vision",
"data mining","data analysis","data science","statistics",
"predictive modeling","feature engineering","time series",
"recommendation systems","bayesian modeling",

# ---------- AI Frameworks ----------
"tensorflow","pytorch","keras","scikit learn","xgboost",
"lightgbm","catboost","huggingface","transformers",

# ---------- Big Data ----------
"spark","pyspark","hadoop","hive","pig","kafka","flink","storm",

# ---------- Databases ----------
"sql","mysql","postgresql","oracle","mongodb","cassandra",
"redis","dynamodb","snowflake","bigquery","redshift",

# ---------- Cloud ----------
"aws","azure","gcp","cloud computing","serverless",
"lambda","ec2","s3","cloud storage","cloud security",

# ---------- DevOps ----------
"docker","kubernetes","jenkins","github actions",
"gitlab ci","terraform","ansible","ci cd",

# ---------- BI ----------
"power bi","tableau","looker","qlik","data visualization",
"dashboarding","reporting","excel","vba",

# ---------- ERP ----------
"sap","sap hana","sap fico","sap mm","sap sd",
"oracle erp","oracle financials","baan","4th shift",

# ---------- Analytics ----------
"business intelligence","etl","elt","data warehousing",
"data governance","data quality","master data",

# ---------- Testing ----------
"unit testing","integration testing","automation testing",
"selenium","cypress","jest","pytest",

# ---------- Security ----------
"cyber security","network security","penetration testing",
"owasp","encryption","iam","zero trust",

# ---------- Mobile ----------
"android","ios","react native","flutter","xamarin",

# ---------- Architecture ----------
"microservices","rest api","graphql","system design",
"distributed systems","event driven architecture",

# ---------- OS / Infra ----------
"linux","unix","windows server","networking","virtualization",

# ---------- Tools ----------
"git","jira","confluence","notion","slack","postman",

# ---------- Soft Skills ----------
"leadership","communication","team management",
"problem solving","critical thinking","decision making",
"stakeholder management","conflict resolution",
"time management","negotiation","mentoring",

# ---------- Business ----------
"supply chain","logistics","inventory management",
"finance","accounting","risk analysis","compliance",
"procurement","operations","project management",

# ---------- Methodologies ----------
"agile","scrum","kanban","waterfall","lean","six sigma",

# ---------- Misc ----------
"technical writing","documentation","research",
"presentation skills","training","coaching"
}


def extract_skills(text):

    found = set()

    words = text.lower()

    for skill in SKILL_SET:
        if skill in words:
            found.add(skill)

    return list(found)



# ---------------- EXPERIENCE YEARS (FLEXIBLE) ----------------
def extract_experience_years(text):

    patterns = [
        r'(\d+)\s+years?\s+of\s+experience',
        r'(\d+)\s+year\s+experience',
        r'(\d+)\+?\s+years?'
    ]

    t = text.lower()

    for p in patterns:
        m = re.search(p, t)
        if m:
            return int(m.group(1))

    return None


# ---------------- MASTER ----------------
def parse_resume(text):

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text)
    }
