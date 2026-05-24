import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import plotly.express as px
import re
from urllib.parse import quote
from transformers import pipeline

# =====================================================
# CONFIG & CUSTOM CSS (Untuk Visual Ala PantauTular)
# =====================================================
st.set_page_config(page_title="PantauTular - Media Intelligence", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    /* Background aplikasi utama */
    .stApp {
        background-color: #F4F7F6;
    }
    
    /* Mempercantik Sidebar / Panel Filter */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
        padding-top: 20px;
    }
    
    /* Custom Card untuk Metrik Informasi */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #2F80ED;
        margin-bottom: 15px;
    }
    .metric-danger { border-left-color: #EB5757; background-color: #FFF5F5; }
    .metric-success { border-left-color: #27AE60; background-color: #F2F9F5; }
    .metric-warning { border-left-color: #F2C94C; background-color: #FFFDF2; }
    
    .metric-title { font-size: 13px; color: #828282; font-weight: 600; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #333333; }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# 1. LOGIKA RELEVANSI V2 (Lebih Fleksibel & Sensitif)
# =====================================================
def hitung_relevansi_v2(keyword, judul, isi):
    if not judul or not keyword:
        return 0
    
    judul_low = judul.lower()
    isi_low = isi.lower()
    keyword_low = keyword.lower()
    
    score = 0
    
    # Perubahan: Menggunakan pencarian substring terarah (bukan regex kata utuh kaku)
    # Ini membuat kata "covid" tetap terdeteksi pada judul "COVID-19" atau "Vaksin Covid"
    if keyword_low in judul_low:
        score += 70
    else:
        # Jika tidak ada kecocokan mutlak, cek potongan kata kunci
        words = keyword_low.split()
        match_count = sum(1 for w in words if w in judul_low)
        if words:
            score += (match_count / len(words)) * 50

    # B. CEK KEMUNCULAN DI ISI ARTIKEL (Bobot 30%)
    isi_matches = isi_low.count(keyword_low)
    if isi_matches >= 2:
        score += 30
    elif isi_matches == 1:
        score += 15
        
    return min(round(score, 1), 100.0)

# =====================================================
# 2. LOAD MODEL MACHINE LEARNING (IndoBERT)
# =====================================================
@st.cache_resource
def load_sentiment_model():
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
        if 'POS' in label or '1' in label: return "Positif"
        if 'NEG' in label or '0' in label: return "Negatif"
        return "Netral"
    except:
        return "Netral"

# =====================================================
# 3. FUNGSI SCRAPER (Limit Ditambahkan Agar Hasil Melimpah)
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
        
        # Perubahan: Menambah batas pengecekan artikel dari 4 menjadi 12 artikel per media
        max_articles_to_check = 12 
        count = 0
        
        for link_tag in links:
            if count >= max_articles_to_check: 
                break
                
            link, title = link_tag["href"], link_tag.get_text(strip=True)
            
            if (link.startswith("http") and portal in link.lower() and len(title) > 30 and link not in seen_links):
                if any(x in link for x in ["/tag/", "/indeks", "/author/"]): 
                    continue
                
                seen_links.add(link)
                isi = get_content(link, portal)
                
                # Hitung Relevansi
                relevansi = hitung_relevansi_v2(keyword, title, isi)
                
                # Perubahan: Ambang batas diturunkan dari 45 ke 30 agar cakupan berita lebih luas
                if relevansi >= 30:
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
                    time.sleep(0.2) # Jeda scraping sedikit dipercepat
    except: 
        pass
    return data

# =====================================================
# 4. TAMPILAN UTAMA ANTARMUKA DASBOR
# =====================================================

# Judul Utama Aplikasi
st.markdown("<h2 style='color: #2F80ED; margin-bottom: 0;'>🤖 PantauTular Media Intelligence</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #828282;'>Sistem Klasifikasi AI & Peta Data Distribusi Berita Penyakit Menular</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout Kontrol Filter di Menu Sidebar Kiri
with st.sidebar:
    st.markdown("<h3 style='color: #2F80ED;'>Filter Informasi</h3>", unsafe_allow_html=True)
    keyword_input = st.text_input("Kata Kunci Penyakit:", "covid")
    
    # Input tambahan estetik untuk menyamakan UI dasbor asli
    st.selectbox("Lokasi Wilayah", ["Semua Lokasi", "DKI Jakarta", "Jawa Barat", "Banten", "Jawa Tengah"])
    st.selectbox("Sumber Portal", ["Semua Portal Berita", "Detik.com", "Kompas.com", "CNN Indonesia", "Republika"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    btn_cari = st.button("Terapkan Filter / Cari AI", use_container_width=True, type="primary")

# Pemrosesan Data & Visualisasi Utama
if btn_cari and keyword_input:
    all_results = []
    portals = ["detik", "kompas", "cnn", "republika"]
    
    msg = st.empty()
    bar = st.progress(0)
    
    for idx, p in enumerate(portals):
        msg.info(f"Mengekstrak data dari portal {p.upper()}...")
        res = crawl_portal(keyword_input, p)
        all_results.extend(res)
        bar.progress((idx + 1) / len(portals))
    
    msg.empty()
    bar.empty()
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        pos_count = len(df[df['Sentimen'] == 'Positif'])
        neg_count = len(df[df['Sentimen'] == 'Negatif'])
        net_count = len(df[df['Sentimen'] == 'Netral'])
        total_art = len(df)

        # Tab Menu Atas (Sesuai dengan Gambar Contoh)
        tab1, tab2 = st.tabs(["Informasi Umum", "Urutan Berita / Detail"])
        
        with tab1:
            st.markdown("### Informasi Hasil Ekstraksi Berita")
            
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📋 TOTAL ARTIKEL LOLOS VERIFIKASI</div>
                    <div class="metric-value">{total_art} Artikel</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card metric-danger">
                    <div class="metric-title">🔴 KASUS SENTIMEN NEGATIF</div>
                    <div class="metric-value">{neg_count} Berita <span style="font-size:15px; font-weight:normal; color:#E74C3C;">({round(neg_count/total_art*100, 1)}%)</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card metric-warning">
                    <div class="metric-title">🟡 KASUS SENTIMEN NETRAL</div>
                    <div class="metric-value">{net_count} Berita <span style="font-size:15px; font-weight:normal; color:#F1C40F;">({round(net_count/total_art*100, 1)}%)</span></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card metric-success">
                    <div class="metric-title">🟢 KASUS SENTIMEN POSITIF</div>
                    <div class="metric-value">{pos_count} Berita <span style="font-size:15px; font-weight:normal; color:#2ECC71;">({round(pos_count/total_art*100, 1)}%)</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                # Blok Estimasi Relevansi
                rerata_rel = round(df['Relevansi'].mean(), 1)
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; height: 93%;">
                    <div class="metric-title">🎯 ESTIMASI RERATA RELEVANSI MODEL</div>
                    <div class="metric-value" style="font-size: 55px; color: #2F80ED; margin-top: 35px;">{rerata_rel}%</div>
                    <p style="color: #828282; font-size: 13px; margin-top: 25px;">Data dihitung real-time berdasarkan pembobotan teks pada judul utama berita kesehatan.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Sesi Grafik Distribusi Visual
            cg1, cg2 = st.columns(2)
            with cg1:
                st.markdown("#### 🌍 Distribusi Sumber Data")
                fig_pie = px.pie(df, names='Portal', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with cg2:
                st.markdown("#### 📊 Grafik Komparasi Sentimen IndoBERT")
                fig_bar = px.bar(df['Sentimen'].value_counts().reset_index(), x='Sentimen', y='count', color='Sentimen',
                                 color_discrete_map={'Positif':'#27AE60', 'Netral':'#F2C94C', 'Negatif':'#EB5757'})
                fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.markdown("### 📰 Detail Berita Terklasifikasi")
            for _, row in df.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: white; padding: 18px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #E0E0E0;">
                        <h4 style="margin:0 0 6px 0; color:#2F80ED;">{row['Judul']}</h4>
                        <p style="font-size:12px; color:#828282; margin-bottom: 12px;">Sumber: <b>{row['Portal']}</b> | <a href="{row['Link']}" target="_blank" style="text-decoration:none; color:#2F80ED;">Kunjungi Tautan Berita ↗</a></p>
                        <p style="font-size:14px; color:#4F4F4F; line-height:1.5;">{row['Isi']}...</p>
                        <hr style="margin: 12px 0; border:0; border-top:1px solid #EEEEEE;">
                        <span style="background-color:#E0F2FE; color:#0369A1; padding:4px 10px; border-radius:4px; font-size:12px; font-weight:bold;">🎯 Skor Relevansi: {row['Relevansi']}%</span>
                        <span style="background-color:#F0FDF4; color:#166534; padding:4px 10px; border-radius:4px; font-size:12px; font-weight:bold; margin-left:10px;">🤖 Klasifikasi AI: {row['Sentimen']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
    else:
        st.error("Hasil pencarian kosong atau di bawah batas relevansi minimum. Coba gunakan kata kunci kesehatan lainnya.")
