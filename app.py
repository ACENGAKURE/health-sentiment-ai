import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import plotly.express as px
from urllib.parse import quote

# Konfigurasi Halaman
st.set_page_config(page_title="Health Monitoring Dashboard", page_icon="📊", layout="wide")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# =====================================================
# FUNGSI ANALISIS & SCRAPER
# =====================================================

def hitung_sentimen(teks):
    # Kamus sederhana Bahasa Indonesia
    positif = ['sembuh', 'turun', 'pencegahan', 'aman', 'berhasil', 'vaksin', 'efektif', 'membaik', 'sehat', 'terkendali']
    negatif = ['wabah', 'meninggal', 'bahaya', 'buruk', 'darurat', 'meningkat', 'kritis', 'sakit', 'gejala', 'terinfeksi']
    
    score = 0
    teks = teks.lower()
    for word in positif:
        if word in teks: score += 1
    for word in negatif:
        if word in teks: score -= 1
        
    if score > 0: return "Positif"
    elif score < 0: return "Negatif"
    else: return "Netral"

def hitung_relevansi(keyword, judul, isi):
    teks_lengkap = (judul + " " + isi).lower()
    keyword_words = keyword.lower().split()
    if not teks_lengkap: return 0
    
    # Menghitung persentase kata kunci yang muncul
    matches = sum(1 for word in keyword_words if word in teks_lengkap)
    score = (matches / len(keyword_words)) * 100
    return round(score, 1)

def get_content(url, portal):
    try:
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        selectors = ["div.detail__body-text", "div.read__content", "div.article__content", "div.detail-text", "article"]
        for s in selectors:
            div = soup.select_one(s)
            if div:
                return " ".join([p.get_text(strip=True) for p in div.find_all("p") if len(p.get_text()) > 20])
        return ""
    except: return ""

def crawl_portal(keyword, portal):
    data = []
    query_encoded = quote(keyword)
    urls = {
        "detik": f"https://www.detik.com/search/searchall?query={query_encoded}",
        "kompas": f"https://search.kompas.com/search/?q={query_encoded}",
        "republika": f"https://republika.co.id/search?q={query_encoded}",
        "merdeka": f"https://www.merdeka.com/search?q={query_encoded}",
        "cnn": f"https://www.cnnindonesia.com/search/?query={query_encoded}",
        "liputan6": f"https://www.liputan6.com/search?q={query_encoded}"
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
            if (link.startswith("http") and portal in link.lower() and len(title) > 35 and link not in seen_links):
                if any(x in link for x in ["/tag/", "/indeks", "/author/"]): continue
                seen_links.add(link)
                isi = get_content(link, portal)
                relevansi = hitung_relevansi(keyword, title, isi)
                if relevansi >= 50: # Hanya ambil yang minimal 50% relevan
                    data.append({
                        "Portal": portal.upper(),
                        "Judul": title,
                        "Link": link,
                        "Isi_Singkat": isi[:400],
                        "Sentimen": hitung_sentimen(title + " " + isi),
                        "Relevansi": relevansi
                    })
                    count += 1
                    time.sleep(0.5)
    except: pass
    return data

# =====================================================
# UI STREAMLIT
# =====================================================

st.title("🏥 Health Media Intelligence Dashboard")
st.markdown("Analisis sentimen dan relevansi berita kesehatan dari berbagai portal media Indonesia.")

with st.sidebar:
    st.header("Konfigurasi")
    keyword_input = st.text_input("Nama Penyakit:", placeholder="Misal: Demam Berdarah")
    btn_cari = st.button("Analisis Sekarang")
    st.divider()
    st.caption("Dashboard ini melakukan crawling otomatis dan analisis teks secara real-time.")

if btn_cari and keyword_input:
    all_results = []
    portals = ["detik", "kompas", "republika", "merdeka", "cnn", "liputan6"]
    
    msg = st.empty()
    bar = st.progress(0)
    
    for idx, p in enumerate(portals):
        msg.info(f"Memindai {p.upper()}...")
        res = crawl_portal(keyword_input, p)
        all_results.extend(res)
        bar.progress((idx + 1) / len(portals))
    
    msg.success(f"Berhasil menarik {len(all_results)} berita. Hasil telah disaring ketat.")
    
    if all_results:
        df = pd.DataFrame(all_results)

        # --- ROW 1: METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Artikel", len(df))
        m2.metric("Sentimen Positif", len(df[df['Sentimen'] == 'Positif']), delta_color="normal")
        m3.metric("Sentimen Negatif", len(df[df['Sentimen'] == 'Negatif']), delta_color="inverse")
        m4.metric("Rata-rata Relevansi", f"{round(df['Relevansi'].mean(), 1)}%")

        # --- ROW 2: CHARTS ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribusi Klaster (Sumber)")
            fig_pie = px.pie(df, names='Portal', hole=0.5, color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.subheader("Tren Sentimen Publik")
            sentiment_counts = df['Sentimen'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            fig_bar = px.bar(sentiment_counts, x='Sentimen', y='Jumlah', color='Sentimen',
                             color_discrete_map={'Positif':'#2ecc71', 'Netral':'#95a5a6', 'Negatif':'#e74c3c'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- ROW 3: DATA LIST ---
        st.subheader("Detail Artikel Terkait")
        for _, row in df.iterrows():
            with st.container(border=True):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"**{row['Judul']}**")
                    st.caption(f"Sumber: {row['Portal']} | [Baca Selengkapnya]({row['Link']})")
                    st.write(row['Isi_Singkat'] + "...")
                with col_b:
                    color = "green" if row['Sentimen'] == "Positif" else "red" if row['Sentimen'] == "Negatif" else "gray"
                    st.markdown(f"**Sentimen:** :{color}[{row['Sentimen']}]")
                    st.write(f"**Relevansi:** {row['Relevansi']}%")
    else:
        st.error("Data tidak ditemukan. Coba gunakan keyword yang lebih umum.")