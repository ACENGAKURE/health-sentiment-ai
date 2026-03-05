import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import plotly.express as px
import re
from urllib.parse import quote
from transformers import pipeline

# Konfigurasi Halaman
st.set_page_config(page_title="IndoBERT Media Intelligence", page_icon="🤖", layout="wide")

# =====================================================
# 1. LOGIKA RELEVANSI V2 (Mencegah Skor 100% Palsu)
# =====================================================
def hitung_relevansi_v2(keyword, judul, isi):
    if not judul or not keyword:
        return 0
    
    judul_low = judul.lower()
    isi_low = isi.lower()
    keyword_words = keyword.lower().split()
    
    score = 0
    total_words = len(keyword_words)
    
    if total_words == 0:
        return 0

    # A. CEK JUDUL (Bobot 70%) - Menggunakan Regex untuk kata utuh
    judul_matches = 0
    for word in keyword_words:
        if re.search(r'\b' + re.escape(word) + r'\b', judul_low):
            judul_matches += 1
    
    score += (judul_matches / total_words) * 70

    # B. CEK ISI (Bobot 30%)
    isi_matches = 0
    for word in keyword_words:
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', isi_low))
        if count >= 2:
            isi_matches += 1
        elif count == 1:
            isi_matches += 0.5
            
    score += (isi_matches / total_words) * 30

    return min(round(score, 1), 100.0)

# =====================================================
# 2. LOAD MODEL MACHINE LEARNING (IndoBERT)
# =====================================================
@st.cache_resource
def load_sentiment_model():
    # Menggunakan model yang lebih stabil untuk menghindari OSError
    try:
        return pipeline("sentiment-analysis", model="w11wo/indobert-sentiment-analysis")
    except:
        return pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p1")

nlp_model = load_sentiment_model()

def hitung_sentimen_ml(teks):
    if not teks: return "Netral"
    try:
        hasil = nlp_model(teks[:512])[0]
        label = hasil['label'].upper()
        # Mapping label model ke kategori manusia
        if 'POS' in label or '1' in label: return "Positif"
        if 'NEG' in label or '0' in label: return "Negatif"
        return "Netral"
    except:
        return "Netral"

# =====================================================
# 3. FUNGSI SCRAPER
# =====================================================
def get_content(url, portal):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        selectors = ["div.detail__body-text", "div.read__content", "div.article__content", "article"]
        for s in selectors:
            div = soup.select_one(s)
            if div:
                return " ".join([p.get_text(strip=True) for p in div.find_all("p") if len(p.get_text()) > 20])
        return ""
    except: return ""

def crawl_portal(keyword, portal):
    data = []
    query_encoded = quote(keyword)
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = {
        "detik": f"https://www.detik.com/search/searchall?query={query_encoded}",
        "kompas": f"https://search.kompas.com/search/?q={query_encoded}",
        "cnn": f"https://www.cnnindonesia.com/search/?query={query_encoded}",
        "republika": f"https://republika.co.id/search?q={query_encoded}"
    }
    
    try:
        r = requests.get(urls[portal], headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=True)
        seen_links = set()
        count = 0
        
        for link_tag in links:
            if count >= 4: break
            link, title = link_tag["href"], link_tag.get_text(strip=True)
            
            if (link.startswith("http") and portal in link.lower() and len(title) > 30 and link not in seen_links):
                if any(x in link for x in ["/tag/", "/indeks", "/author/"]): continue
                
                seen_links.add(link)
                isi = get_content(link, portal)
                
                # Hitung Relevansi & Sentimen
                relevansi = hitung_relevansi_v2(keyword, title, isi)
                
                # Filter ketat: Hanya tampilkan yang benar-benar nyambung
                if relevansi >= 45:
                    sentimen = hitung_sentimen_ml(title + " " + isi)
                    data.append({
                        "Portal": portal.upper(),
                        "Judul": title,
                        "Link": link,
                        "Isi": isi[:300],
                        "Sentimen": sentimen,
                        "Relevansi": relevansi
                    })
                    count += 1
                    time.sleep(0.5)
    except: pass
    return data

# =====================================================
# 4. UI DASHBOARD
# =====================================================
st.title("📊 Health Media Intelligence Dashboard")
st.markdown("Implementasi Machine Learning **IndoBERT** untuk analisis sentimen berita kesehatan.")

with st.sidebar:
    st.header("Pencarian")
    keyword_input = st.text_input("Kata Kunci:", "Demam Berdarah")
    btn_cari = st.button("Mulai Analisis AI")
    st.divider()
    st.write("Sistem ini menyaring berita berdasarkan bobot judul (70%) dan isi (30%).")

if btn_cari and keyword_input:
    all_results = []
    portals = ["detik", "kompas", "cnn", "republika"]
    
    msg = st.empty()
    bar = st.progress(0)
    
    for idx, p in enumerate(portals):
        msg.info(f"Menganalisis {p.upper()}...")
        res = crawl_portal(keyword_input, p)
        all_results.extend(res)
        bar.progress((idx + 1) / len(portals))
    
    msg.success(f"Analisis Selesai! Menemukan {len(all_results)} berita.")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # ROW 1: METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Artikel", len(df))
        m2.metric("Sentimen Positif", len(df[df['Sentimen'] == 'Positif']))
        m3.metric("Sentimen Negatif", len(df[df['Sentimen'] == 'Negatif']))
        m4.metric("Rerata Relevansi", f"{round(df['Relevansi'].mean(), 1)}%")

        # ROW 2: CHARTS
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribusi Sumber")
            st.plotly_chart(px.pie(df, names='Portal', hole=0.4), use_container_width=True)
            
        with c2:
            st.subheader("Analisis Sentimen AI")
            fig_bar = px.bar(df['Sentimen'].value_counts().reset_index(), x='Sentimen', y='count', color='Sentimen',
                             color_discrete_map={'Positif':'#2ecc71', 'Netral':'#95a5a6', 'Negatif':'#e74c3c'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # ROW 3: LIST
        st.subheader("Detail Berita Terverifikasi")
        for _, row in df.iterrows():
            with st.container(border=True):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"**{row['Judul']}**")
                    st.caption(f"Sumber: {row['Portal']} | [Link Berita]({row['Link']})")
                    st.write(row['Isi'] + "...")
                with col_b:
                    st.write(f"**Relevansi:** {row['Relevansi']}%")
                    st.write(f"**Sentimen:** {row['Sentimen']}")
    else:
        st.error("Hasil tidak ditemukan. Pastikan kata kunci spesifik.")
