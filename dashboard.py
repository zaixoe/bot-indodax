# ==============================================================================
#           APEX QUANTUM ANALYTICS TERMINAL (AQAT) - v1.1
# ==============================================================================
# SEBUAH DASHBOARD STREAMLIT UNTUK ANALISIS PERFORMA TRADING BOT TINGKAT LANJUT
# FITUR: Metrik Kuantitatif, Visualisasi Drawdown, Filter Dinamis, Laporan PDF,
#        Penanganan 'Empty State' yang Profesional.
# ==============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fpdf import FPDF

# --- [1] KONFIGURASI HALAMAN & GAYA ---
st.set_page_config(
    layout="wide",
    page_title="Apex Crypto Analytics",
    page_icon="🤖"
)
# Sembunyikan footer default Streamlit dan menu hamburger
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- [2] KONFIGURASI KONEKSI API ---
BASE_URL = st.secrets.get("BASE_URL", "http://127.0.0.1:5000/api/v1/dashboard")
API_KEY = st.secrets.get("DASHBOARD_API_KEY", "your-fallback-key")
HEADERS = {"X-API-Key": API_KEY}


# --- [3] FUNGSI-FUNGSI PENGOLAHAN DATA & ANALITIK ---

@st.cache_data(ttl=60)
def get_trades_data():
    """Mengambil semua data trade dari backend API dan memprosesnya menjadi DataFrame."""
    try:
        response = requests.get(f"{BASE_URL}/trades", headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'], errors='coerce')
        df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'], errors='coerce')
        numeric_cols = ['entry_price', 'exit_price', 'quantity', 'modal', 'pnl_percent']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_values(by='entry_timestamp', ascending=False)
        
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal terhubung ke API Bot: {e}")
        return pd.DataFrame()
    
@st.cache_data(ttl=30)
def get_signals_data():
    """Mengambil data sinyal terbaru dari backend API."""
    try:
        # [PERBAIKAN] Tambahkan timeout yang wajar (misal, 15 detik)
        response = requests.get(f"{BASE_URL}/signals", headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        # Jangan tampilkan error besar, cukup log di terminal
        print(f"Gagal mengambil data sinyal: {e}")
        return pd.DataFrame()

# GANTI FUNGSI INI DENGAN VERSI YANG SUDAH DIBERSIHKAN

@st.cache_data
def calculate_advanced_metrics(_df):
    """Menghitung metrik statistik kuantitatif dari DataFrame trade."""
    
    # Penjaga gerbang utama untuk fungsi ini
    required_cols = ['status', 'pnl_percent']
    if _df.empty or not all(col in _df.columns for col in required_cols):
        return {
            "total_pnl_percent": 0, "total_trades": 0, "win_rate_percent": 0,
            "profit_factor": 0, "sharpe_ratio": 0, "max_drawdown_percent": 0,
            "expectancy_percent": 0, "winning_trades": 0, "losing_trades": 0
        }

    # Lanjutkan ke logika utama
    df = _df[_df['status'] == 'closed'].dropna(subset=['pnl_percent'])
    
    # Penjaga kedua untuk kasus khusus (misal, kurang dari 2 trade)
    if df.empty or len(df) < 2:
        # Menangani kasus jika hanya ada 0 atau 1 trade
        total_pnl = df['pnl_percent'].sum() if not df.empty else 0
        winning_trades_count = len(df[df['pnl_percent'] > 0])
        return {
            "total_pnl_percent": total_pnl,
            "total_trades": len(df),
            "winning_trades": winning_trades_count,
            "losing_trades": len(df) - winning_trades_count,
            "win_rate_percent": 100 if len(df) > 0 and winning_trades_count == len(df) else 0,
            "profit_factor": float('inf') if total_pnl > 0 else 0,
            "sharpe_ratio": 0, "max_drawdown_percent": 0,
            "expectancy_percent": df['pnl_percent'].mean() if not df.empty else 0
        }

    # Kalkulasi penuh jika ada cukup data
    total_trades = len(df)
    pnl = df['pnl_percent']
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    win_rate = (len(wins) / total_trades * 100)
    total_profit = wins.sum(); total_loss = abs(losses.sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_return = pnl.mean(); std_return = pnl.std()
    sharpe_ratio = (avg_return / std_return) * np.sqrt(365) if std_return > 0 else 0
    
    df_sorted = df.sort_values(by='exit_timestamp')
    cumulative_pnl = df_sorted['pnl_percent'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max() if not drawdown.empty else 0
    
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0
    expectancy = ((win_rate / 100) * avg_win) - ((1 - win_rate / 100) * avg_loss)
    
    return {
        "total_pnl_percent": total_profit - total_loss, "total_trades": total_trades,
        "winning_trades": len(wins), "losing_trades": len(losses), "win_rate_percent": win_rate,
        "profit_factor": profit_factor, "sharpe_ratio": sharpe_ratio,
        "max_drawdown_percent": max_drawdown, "expectancy_percent": expectancy
    }

@st.cache_data
def generate_pdf_report(_df, _metrics, date_range):
    """Membuat laporan PDF profesional."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Apex Trading Bot - Performance Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Periode: {date_range[0].strftime('%Y-%m-%d')} hingga {date_range[1].strftime('%Y-%m-%d')}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Key Performance Indicators", 0, 1)
    pdf.set_font("Arial", '', 10)
    metrics_to_show = {
        "Total Net P/L (%)": f"{_metrics['total_pnl_percent']:.2f}%", "Win Rate (%)": f"{_metrics['win_rate_percent']:.2f}%",
        "Profit Factor": f"{_metrics['profit_factor']:.2f}", "Sharpe Ratio (Annualized)": f"{_metrics['sharpe_ratio']:.2f}",
        "Maximum Drawdown (%)": f"{_metrics['max_drawdown_percent']:.2f}%", "Total Trades": str(_metrics['total_trades']),
    }
    for key, value in metrics_to_show.items():
        pdf.cell(95, 8, f"  {key}:", 'B', 0)
        pdf.cell(95, 8, value, 'B', 1, 'R')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top 5 & Bottom 5 Trades", 0, 1)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(40, 6, 'Pair', 1, 0, 'C'); pdf.cell(40, 6, 'Strategy', 1, 0, 'C');
    pdf.cell(40, 6, 'Entry Date', 1, 0, 'C'); pdf.cell(30, 6, 'P/L (%)', 1, 1, 'C')
    pdf.set_font("Arial", '', 8)
    closed_trades = _df[_df['status'] == 'closed'].sort_values('pnl_percent', ascending=False).dropna(subset=['entry_timestamp', 'pnl_percent'])
    for _, row in pd.concat([closed_trades.head(5), closed_trades.tail(5)]).iterrows():
        pdf.cell(40, 5, str(row['pair']), 1); pdf.cell(40, 5, str(row['strategy_type']), 1)
        pdf.cell(40, 5, row['entry_timestamp'].strftime('%Y-%m-%d'), 1); pdf.cell(30, 5, f"{row['pnl_percent']:.2f}%", 1, 1, 'R')
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "This report is generated for analytical purposes and does not constitute investment advice.", 0, 1, 'C')
    return pdf.output(dest='S').encode('latin-1')


# --- [4] TAMPILAN UTAMA DASHBOARD ---
st.title("Apex Crypto Signal Analytics Terminal")
st.markdown("---")

# --- Konten Utama ---
signals_df = get_signals_data()
master_df = get_trades_data()

# --- [FITUR BARU] Tampilkan Live Signal Radar ---
st.header("📡 Live Signal Radar")
if not signals_df.empty:
    st.dataframe(signals_df, use_container_width=True)
else:
    st.info("Saat ini tidak ada sinyal trading aktif yang terdeteksi. Bot sedang memantau pasar...")

st.markdown("---")
# --- AKHIR FITUR BARU ---

# --- Sidebar ---
st.sidebar.header("⚙️ Filter Analisis")
if not master_df.empty:
    date_range = st.sidebar.date_input(
        "Pilih Rentang Waktu",
        value=(master_df['entry_timestamp'].min().date(), datetime.now().date()),
        min_value=master_df['entry_timestamp'].min().date(), max_value=datetime.now().date(),
    )
    start_date = datetime.combine(date_range[0], datetime.min.time())
    end_date = datetime.combine(date_range[1], datetime.max.time())
    
    selected_pairs = st.sidebar.multiselect("Pilih Aset", sorted(master_df['pair'].unique()), default=sorted(master_df['pair'].unique()))
    selected_strategies = st.sidebar.multiselect("Pilih Strategi", sorted(master_df['strategy_type'].unique()), default=sorted(master_df['strategy_type'].unique()))
    
    filtered_df = master_df[
        (master_df['entry_timestamp'] >= start_date) & (master_df['entry_timestamp'] <= end_date) &
        (master_df['pair'].isin(selected_pairs)) & (master_df['strategy_type'].isin(selected_strategies))
    ].copy()
else:
    st.sidebar.info("Filter akan aktif setelah ada data transaksi.")
    filtered_df = master_df

# --- Konten Utama ---
if filtered_df.empty and master_df.empty:
    st.warning("Menunggu data transaksi pertama dari bot...")
elif filtered_df.empty and not master_df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih.")

# Kalkulasi Metrik (selalu jalankan, akan menghasilkan nol jika kosong)
metrics = calculate_advanced_metrics(filtered_df)

# Tampilan Metrik Utama (selalu tampilkan)
st.header("Key Performance Indicators (KPIs)")
cols = st.columns(4)
cols[0].metric("Total Net P/L (%)", f"{metrics['total_pnl_percent']:.2f}%")
cols[1].metric("Win Rate", f"{metrics['win_rate_percent']:.1f}%")
cols[2].metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
cols[3].metric("Sharpe Ratio (Ann.)", f"{metrics['sharpe_ratio']:.2f}")
cols = st.columns(4)
cols[0].metric("Maximum Drawdown (%)", f"{metrics['max_drawdown_percent']:.2f}%")
cols[1].metric("Expectancy per Trade (%)", f"{metrics['expectancy_percent']:.2f}%")
cols[2].metric("Total Trades", metrics['total_trades'])
cols[3].metric("Menang / Kalah", f"{metrics['winning_trades']} / {metrics['losing_trades']}")

# Tombol Ekspor (hanya tampilkan jika ada data)
if not filtered_df.empty:
    pdf_data = generate_pdf_report(filtered_df, metrics, date_range)
    st.sidebar.download_button(
        label="📄 Unduh Laporan PDF", data=pdf_data,
        file_name=f"apex_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.header("Visualisasi Performa")
tab1, tab2, tab3 = st.tabs(["📈 Kurva Ekuitas", "📊 Analisis per Aset", "📜 Detail Transaksi"])

with tab1:
    if not filtered_df.empty and 'status' in filtered_df.columns:
        equity_df = filtered_df[filtered_df['status'] == 'closed'].copy()
        if not equity_df.empty:
            equity_df.sort_values(by='exit_timestamp', inplace=True)
            equity_df['cumulative_pnl'] = equity_df['pnl_percent'].cumsum()
            equity_df['running_max'] = equity_df['cumulative_pnl'].cummax()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_df['exit_timestamp'], y=equity_df['running_max'], fill=None, mode='lines', line_color='rgba(255,255,255,0)'))
            fig.add_trace(go.Scatter(x=equity_df['exit_timestamp'], y=equity_df['cumulative_pnl'], fill='tonexty', mode='lines', line_color='cyan', name='Equity Curve'))
            fig.update_layout(title_text='Kurva Pertumbuhan Ekuitas dengan Periode Drawdown (Area Abu-abu)', xaxis_title='Tanggal', yaxis_title='Total P/L Kumulatif (%)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Menunggu data transaksi yang ditutup untuk menampilkan Kurva Ekuitas.")
    else:
        st.info("Tidak ada data yang cocok dengan filter untuk menampilkan Kurva Ekuitas.")

with tab2:
    if not filtered_df.empty and 'status' in filtered_df.columns:
        pnl_by_pair_df = filtered_df[filtered_df['status'] == 'closed'].copy()
        if not pnl_by_pair_df.empty:
            pnl_by_pair = pnl_by_pair_df.groupby('pair')['pnl_percent'].sum().sort_values(ascending=False)
            fig_pair_pnl = px.bar(pnl_by_pair, title="Total P/L (%) per Aset")
            st.plotly_chart(fig_pair_pnl, use_container_width=True)
        else:
            st.info("Menunggu data transaksi yang ditutup untuk menampilkan Analisis per Aset.")
    else:
        st.info("Tidak ada data yang cocok dengan filter untuk menampilkan Analisis per Aset.")
        
with tab3:
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.info("Tidak ada data yang cocok dengan filter untuk menampilkan Detail Transaksi.")
