# ==============================================================================
#           APEX QUANTUM ANALYTICS TERMINAL (AQAT) - v2.0 (Live First)
# ==============================================================================
# VERSI INI MEMISAHKAN TAMPILAN POSISI TERBUKA DAN ANALISIS HISTORIS
# UNTUK MEMBERIKAN GAMBARAN LIVE PORTFOLIO DAN PERFORMA MASA LALU SECARA JELAS
# ==============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import math

# --- [1] KONFIGURASI HALAMAN & GAYA ---
st.set_page_config(layout="wide", page_title="Apex Quantum Analytics", page_icon="🤖")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- [2] KONFIGURASI KONEKSI API ---
BASE_URL = st.secrets.get("BASE_URL", "http://127.0.0.1:5000/api/v1/dashboard")
API_KEY = st.secrets.get("DASHBOARD_API_KEY", "your-fallback-key")
HEADERS = {"X-API-Key": API_KEY}

# --- [3] FUNGSI-FUNGSI PENGOLAHAN DATA & ANALITIK ---

# [PERBAIKAN] Mengurangi TTL agar data lebih fresh
@st.cache_data(ttl=10)
def get_trades_data(endpoint):
    """Fungsi untuk mengambil data historis trade."""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        for col in ['entry_timestamp', 'exit_timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Gagal terhubung ke API Bot: {e}. Pastikan bot berjalan dan BASE_URL sudah benar.")
        return pd.DataFrame()

# Fungsi calculate_advanced_metrics dan generate_pdf_report tetap sama
# (Tidak perlu diubah)
@st.cache_data
def calculate_advanced_metrics(_df):
    if _df.empty or 'status' not in _df.columns or 'pnl_percent' not in _df.columns:
        return {"total_pnl_percent": 0, "total_trades": 0, "win_rate_percent": 0, "profit_factor": 0,
                "sharpe_ratio": 0, "max_drawdown_percent": 0, "expectancy_percent": 0,
                "winning_trades": 0, "losing_trades": 0}

    df = _df[_df['status'] == 'closed'].copy()
    df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce')
    df.dropna(subset=['pnl_percent'], inplace=True)
    df = df[df['pnl_percent'] > -100.1]

    if df.empty:
        return {"total_pnl_percent": 0, "total_trades": 0, "win_rate_percent": 0, "profit_factor": 0,
                "sharpe_ratio": 0, "max_drawdown_percent": 0, "expectancy_percent": 0,
                "winning_trades": 0, "losing_trades": 0}

    total_trades = len(df)
    pnl = df['pnl_percent']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    
    winning_trades_count = len(wins)
    losing_trades_count = len(losses)
    
    win_rate = (winning_trades_count / total_trades * 100) if total_trades > 0 else 0
    total_profit = wins.sum()
    total_loss = abs(losses.sum())
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_return = pnl.mean()
    std_return = pnl.std()
    sharpe_ratio = (avg_return / std_return) * np.sqrt(365) if std_return is not None and std_return > 0 else 0
    
    max_drawdown = 0
    if 'exit_timestamp' in df.columns and df['exit_timestamp'].notna().all():
        df.sort_values(by='exit_timestamp', inplace=True)
        df['cumulative_pnl'] = df['pnl_percent'].cumsum()
        running_max = df['cumulative_pnl'].cummax()
        drawdown = running_max - df['cumulative_pnl']
        max_drawdown = drawdown.max() if not drawdown.empty else 0
    
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0
    expectancy = ((win_rate / 100) * avg_win) - ((100 - win_rate) / 100 * avg_loss) if total_trades > 0 else 0
    
    return {"total_pnl_percent": pnl.sum(), "total_trades": total_trades,
            "winning_trades": winning_trades_count, "losing_trades": losing_trades_count,
            "win_rate_percent": win_rate, "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio, "max_drawdown_percent": max_drawdown,
            "expectancy_percent": expectancy}
            
@st.cache_data
def generate_pdf_report(_df, _metrics, date_range):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Apex Trading Bot - Performance Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        pdf.cell(0, 10, f"Periode: {date_range[0].strftime('%Y-%m-%d')} hingga {date_range[1].strftime('%Y-%m-%d')}", 0, 1, 'C')
    
    pdf.ln(5)
    # ... (sisa fungsi PDF tetap sama) ...
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Key Performance Indicators", 0, 1)
    pdf.set_font("Arial", '', 10)
    
    profit_factor_display = f"{_metrics['profit_factor']:.2f}" if not math.isinf(_metrics['profit_factor']) else "∞ (No Losses)"
    
    metrics_to_show = {"Total Net P/L (%)": f"{_metrics['total_pnl_percent']:.2f}%", "Win Rate (%)": f"{_metrics['win_rate_percent']:.2f}%",
                       "Profit Factor": profit_factor_display, "Sharpe Ratio (Annualized)": f"{_metrics['sharpe_ratio']:.2f}",
                       "Maximum Drawdown (%)": f"{_metrics['max_drawdown_percent']:.2f}%", "Total Trades": str(_metrics['total_trades'])}
    for key, value in metrics_to_show.items():
        pdf.cell(95, 7, f"  {key}:", 'B', 0); pdf.cell(95, 7, value, 'B', 1, 'R')
    
    pdf.ln(5)
    # ... (sisa fungsi PDF tetap sama) ...
    return pdf.output(dest='S').encode('latin-1')

# --- [4] TAMPILAN UTAMA DASHBOARD ---
st.title("Apex Trading Analytics Terminal")

# Tombol refresh manual untuk menghapus cache
if st.button("🔄 Refresh Data Sekarang"):
    st.cache_data.clear()

master_df = get_trades_data("trades")

# --- [BAGIAN BARU: STATUS LIVE PORTFOLIO] ---
st.header("🛡️ Posisi Terbuka Saat Ini")
if not master_df.empty:
    open_positions_df = master_df[master_df['status'] == 'open'].copy()
    if not open_positions_df.empty:
        # Tampilkan kolom yang paling relevan untuk posisi terbuka
        display_cols = ['pair', 'entry_timestamp', 'entry_price', 'quantity', 'modal', 'strategy_type']
        st.dataframe(open_positions_df[display_cols], use_container_width=True)
    else:
        st.info("Tidak ada posisi yang sedang terbuka saat ini.")
else:
    st.info("Menunggu data dari bot...")

st.markdown("---")

# --- [BAGIAN YANG DIPERBAIKI: ANALISIS HISTORIS] ---
st.header("📊 Analisis Performa Historis (Transaksi Selesai)")
st.caption("Metrik di bawah ini hanya dihitung dari transaksi yang sudah ditutup (`closed`) untuk mengukur efektivitas strategi secara historis.")

with st.expander("⚙️ Filter Analisis & Opsi Laporan", expanded=True):
    if not master_df.empty and 'entry_timestamp' in master_df.columns:
        # Filter hanya berdasarkan data yang sudah ditutup
        closed_df_for_filter = master_df[master_df['status'] == 'closed']
        
        if not closed_df_for_filter.empty:
            min_date = closed_df_for_filter['entry_timestamp'].min().date()
            max_date = datetime.now().date()
            date_range = st.date_input("Pilih Rentang Waktu", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            
            col2, col3 = st.columns(2)
            with col2:
                all_pairs = sorted(closed_df_for_filter['pair'].unique())
                selected_pairs = st.multiselect("Pilih Aset", all_pairs, default=all_pairs)
            with col3:
                all_strategies = sorted(closed_df_for_filter['strategy_type'].dropna().unique())
                selected_strategies = st.multiselect("Pilih Strategi", all_strategies, default=all_strategies)

            if len(date_range) == 2:
                start_date = datetime.combine(date_range[0], datetime.min.time())
                end_date = datetime.combine(date_range[1], datetime.max.time())
                
                # Aplikasikan filter ke dataframe yang sudah difilter 'closed'
                filtered_df = closed_df_for_filter[
                    (closed_df_for_filter['entry_timestamp'] >= start_date) & 
                    (closed_df_for_filter['entry_timestamp'] <= end_date) &
                    (closed_df_for_filter['pair'].isin(selected_pairs)) & 
                    (closed_df_for_filter['strategy_type'].isin(selected_strategies))
                ].copy()
            else:
                filtered_df = pd.DataFrame()
        else:
             st.info("Filter akan aktif setelah ada transaksi yang ditutup.")
             filtered_df = pd.DataFrame()
    else:
        st.info("Filter akan aktif setelah ada data transaksi.")
        filtered_df = pd.DataFrame()

# KPI, Visualisasi, dan Detail Transaksi sekarang menggunakan `filtered_df` yang sudah pasti `closed`
if not filtered_df.empty:
    metrics = calculate_advanced_metrics(filtered_df) # Fungsi ini sudah benar karena menerima data yg sudah difilter
    
    st.subheader("Key Performance Indicators (KPIs)")
    cols = st.columns(4)
    cols[0].metric("Total Net P/L (%)", f"{metrics['total_pnl_percent']:.2f}%")
    # ... (sisa metrik tetap sama) ...
    cols[1].metric("Win Rate", f"{metrics['win_rate_percent']:.1f}%")
    cols[2].metric("Profit Factor", f"{metrics['profit_factor']:.2f}" if not math.isinf(metrics['profit_factor']) else "∞")
    cols[3].metric("Sharpe Ratio (Ann.)", f"{metrics['sharpe_ratio']:.2f}")
    
    cols_b = st.columns(4)
    cols_b[0].metric("Maximum Drawdown (%)", f"{metrics['max_drawdown_percent']:.2f}%")
    cols_b[1].metric("Expectancy per Trade (%)", f"{metrics['expectancy_percent']:.2f}%")
    cols_b[2].metric("Total Trades", metrics['total_trades'])
    cols_b[3].metric("Menang / Kalah", f"{metrics['winning_trades']} / {metrics['losing_trades']}")

    if 'date_range' in locals() and len(date_range) == 2:
        pdf_data = generate_pdf_report(filtered_df, metrics, date_range)
        st.download_button(
            label="📄 Unduh Laporan PDF", data=pdf_data,
            file_name=f"apex_report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf"
        )

    st.markdown("---")
    st.subheader("Visualisasi Performa")
    
    tab1, tab2, tab3 = st.tabs(["📈 Kurva Ekuitas", "📊 Analisis per Aset", "📜 Seluruh Riwayat Transaksi"])
    with tab1:
        # Kurva Ekuitas (logika tetap sama, karena hanya relevan untuk trade yg sudah ditutup)
        equity_df = filtered_df.copy()
        equity_df.sort_values(by='exit_timestamp', inplace=True)
        equity_df['cumulative_pnl'] = equity_df['pnl_percent'].cumsum()
        # ... (sisa kode chart tetap sama) ...
        fig = px.line(equity_df, x='exit_timestamp', y='cumulative_pnl', title='Kurva Pertumbuhan Ekuitas')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Analisis per Aset (logika tetap sama)
        pnl_by_pair = filtered_df.groupby('pair')['pnl_percent'].sum().sort_values(ascending=False)
        fig_pair_pnl = px.bar(pnl_by_pair, title="Total P/L (%) per Aset")
        st.plotly_chart(fig_pair_pnl, use_container_width=True)
        
    with tab3:
        # [PERBAIKAN] Tab ini sekarang menampilkan SEMUA transaksi (terbuka dan tertutup)
        st.info("Tabel di bawah ini menampilkan seluruh riwayat transaksi, termasuk yang masih terbuka.")
        st.dataframe(master_df, use_container_width=True) # Gunakan master_df, 
