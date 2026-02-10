import streamlit as st
import pandas as pd
import requests
import os
import datetime
import time
import io
from dotenv import load_dotenv

# --- AUTH & AI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.credentials import Credentials

# --- FILE READERS ---
try:
    from pypdf import PdfReader
except ImportError: PdfReader = None
try:
    from docx import Document
except ImportError: Document = None

# ==========================================
# 🔐 CONFIG & AUTH SETUP
# ==========================================
load_dotenv() # Load variables from .env file

def get_secret(key):
    # Priority 1: Check Environment Variables (Railway / .env)
    # This prevents the "SecretNotFoundError" crash on Railway
    if key in os.environ:
        return os.environ[key]

    # Priority 2: Check Streamlit Secrets (Local / Streamlit Cloud)
    # We wrap this in a try/except so it doesn't crash if secrets.toml is missing
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    return None

# Fetch Secrets using the safe function
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
CLIENT_ID = get_secret("GOOGLE_CLIENT_ID")
CLIENT_SECRET = get_secret("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = get_secret("REDIRECT_URI") 

SCOPES = ['https://www.googleapis.com/auth/drive.file'] 
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# ==========================================
# 🔑 OAUTH 2.0 LOGIN FLOW
# ==========================================
def authorize_google():
    """Handles the 3-legged OAuth flow."""
    if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
        st.error("⚠️ Missing Google OAuth Secrets in Railway Variables.")
        return None

    # Configuration dictionary for the Flow
    client_config = {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    # Create the Flow instance
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    # 1. Check if we have an Authorization Code from Google (in the URL)
    if "code" in st.query_params:
        code = st.query_params["code"]
        try:
            flow.fetch_token(code=code)
            credentials = flow.credentials
            st.session_state["google_creds"] = {
                "token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_uri": credentials.token_uri,
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scopes": credentials.scopes
            }
            # Clear the URL code so we don't re-trigger
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

    # 2. Return Credentials if logged in
    if "google_creds" in st.session_state:
        return Credentials(**st.session_state["google_creds"])
    
    # 3. If NOT logged in, show Login Button
    return None

def get_login_url():
    if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
        return "#"
        
    client_config = {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return auth_url

# ==========================================
# 📂 CORE LOGIC (READERS & AI)
# ==========================================
def get_text_from_upload(uploaded_file):
    try:
        if uploaded_file.name.endswith('.pdf'):
            if PdfReader is None: return "Error: Install pypdf"
            reader = PdfReader(uploaded_file)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif uploaded_file.name.endswith('.docx'):
            if Document is None: return "Error: Install python-docx"
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e: return f"Error: {e}"

def fetch_recent_trials(disease_query, months_back=6):
    today = datetime.date.today()
    start_date_threshold = today - datetime.timedelta(days=30*months_back)
    params = {
        "query.term": disease_query,       
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
        "pageSize": 300,                   
        "sort": "StudyFirstPostDate:desc"  
    }
    try:
        headers = {"User-Agent": "ResearchAgent/OAuth"}
        response = requests.get(CLINICAL_TRIALS_API, params=params, headers=headers)
        if response.status_code != 200: return []
        data = response.json()
    except: return []
    
    studies = []
    if 'studies' in data:
        for study in data['studies']:
            protocol = study.get('protocolSection', {})
            id_mod = protocol.get('identificationModule', {})
            stat_mod = protocol.get('statusModule', {})
            cond_mod = protocol.get('conditionsModule', {})
            
            start_date_str = stat_mod.get('startDateStruct', {}).get('date')
            if not start_date_str: continue

            try:
                if len(start_date_str) == 7:
                    dt = datetime.datetime.strptime(start_date_str, "%Y-%m").date()
                else:
                    dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
                if dt < start_date_threshold: continue
            except: continue

            matches = [c for c in cond_mod.get('conditions', []) if disease_query.lower() in c.lower()]
            if not matches: continue 

            studies.append({
                "NCT Number": id_mod.get('nctId'),
                "Study title": id_mod.get('briefTitle'),
                "BriefSummary": protocol.get('descriptionModule', {}).get('briefSummary', ''),
                "Condition": ", ".join(matches),
                "Status": stat_mod.get('overallStatus'),
                "Date": start_date_str
            })
    return studies

def analyze_trials(research_text, trials, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)
    
    # 1. Summarize User Paper
    with st.status("Reading your document..."):
        summary_chain = ChatPromptTemplate.from_template("Summarize goals in 3 sentences: {text}") | llm
        research_summary = summary_chain.invoke({"text": research_text[:25000]}).content
        st.write(f"**Context:** {research_summary}")

    # 2. Batch Analysis
    analyzed_data = []
    BATCH_SIZE = 20
    progress = st.progress(0)
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    MY RESEARCH: {research_summary}
    TRIALS: {trials_text}
    OUTPUT: Trial_ID|Relevance(Yes/No)|Reason|Update Potential
    """)

    for i in range(0, len(trials), BATCH_SIZE):
        batch = trials[i:i+BATCH_SIZE]
        batch_txt = "\n".join([f"ID: {t['NCT Number']}\nTitle: {t['Study title']}\nSum: {t['BriefSummary'][:500]}\n---" for t in batch])
        
        try:
            res = (analysis_prompt | llm).invoke({"research_summary": research_summary, "trials_text": batch_txt})
            res_map = {l.split('|')[0].strip(): l.split('|')[1:] for l in res.content.split('\n') if '|' in l}
            
            for t in batch:
                if t['NCT Number'] in res_map and len(res_map[t['NCT Number']]) >= 3:
                    vals = res_map[t['NCT Number']]
                    t['AI Relevance'], t['AI Reason'], t['AI Potential'] = vals[0].strip(), vals[1].strip(), vals[2].strip()
                else:
                    t['AI Relevance'] = "No"
                if 'BriefSummary' in t: del t['BriefSummary']
                analyzed_data.append(t)
        except: pass
        progress.progress(min((i+BATCH_SIZE)/len(trials), 1.0))

    return pd.DataFrame(analyzed_data)

def upload_to_drive(creds, df, filename):
    try:
        service = build('drive', 'v3', credentials=creds)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        file_meta = {'name': filename, 'mimeType': 'text/csv'}
        media = MediaIoBaseUpload(csv_buffer, mimetype='text/csv')
        
        file = service.files().create(body=file_meta, media_body=media, fields='id, webViewLink').execute()
        return file.get('webViewLink')
    except Exception as e:
        st.error(f"Upload Error: {e}")
        return None

# ==========================================
# 🚀 MAIN APP
# ==========================================
def main():
    st.set_page_config(page_title="Rajah", layout="wide")
    
    # --- AUTH SECTION ---
    creds = authorize_google()
    
    with st.sidebar:
        st.title("🧬 Rajah - Research Agent")
        
        if not creds:
            st.warning("⚠️ Login to enable Auto-Save")
            login_url = get_login_url()
            st.link_button("Login with Google", login_url)
        else:
            st.success("✅ Logged into Drive")
            if st.button("Logout"):
                del st.session_state["google_creds"]
                st.rerun()

        st.divider()
        st.header("1. Upload")
        uploaded_file = st.file_uploader("Research Paper", type=['pdf', 'docx', 'txt'])
        
        st.header("2. Settings")
        disease = st.text_input("Disease", "Rheumatoid Arthritis")
        months = st.slider("Months Back", 1, 12, 6)
        run_btn = st.button("Start Analysis", type="primary")

    # --- MAIN EXECUTION ---
    if run_btn and uploaded_file:
        if not GEMINI_API_KEY or "PASTE" in GEMINI_API_KEY:
            st.error("Missing Gemini API Key in Variables/Secrets!")
            return

        text = get_text_from_upload(uploaded_file)
        if "Error" in text:
            st.error(text)
            return

        trials = fetch_recent_trials(disease, months)
        
        if trials:
            st.info(f"Found {len(trials)} trials. Analyzing...")
            df = analyze_trials(text, trials, GEMINI_API_KEY)
            
            if not df.empty:
                relevant = df[df['AI Relevance'].str.contains("Yes", case=False, na=False)]
                st.subheader("Results")
                st.dataframe(relevant, use_container_width=True)
                
                # --- SAVE OPTIONS ---
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "updates.csv", "text/csv")
                
                with col2:
                    if creds:
                        if st.button("☁️ Save to My Google Drive"):
                            link = upload_to_drive(creds, relevant, f"Updates_{disease}.csv")
                            if link: st.success(f"Saved! [View File]({link})")
                    else:
                        st.caption("Login to save directly to Drive.")
        else:
            st.warning("No trials found.")

if __name__ == "__main__":
    main()
