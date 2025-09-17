# ==============================================================================
#           APEX QUANTUM ANALYTICS TERMINAL (AQAT) - v1.2
# ==============================================================================
# SEBUAH DASHBOARD STREAMLIT UNTUK ANALISIS PERFORMA TRADING BOT TINGKAT LANJUT
# FITUR: Live Signal Analysis, What-If Simulation, Quantitative Metrics,
#        Calendar Heatmap, Drawdown Visualization, PDF Reports.
# ==============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calmap
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- [1] KONFIGURASI HALAMAN & GAYA ---
st.set_page_config(layout="wide", page_title="Apex Trading Analytics", page_icon="🤖")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- [2] KONFIGURASI KONEKSI API ---
BASE_URL = st.secrets.get("BASE_URL", "http://127.0.0.1:5000/api/v1/dashboard")
API_KEY = st.secrets.get("DASHBOARD_API_KEY", "your-fallback-key")
HEADERS = {"X-API-Key": API_KEY}

# --- [3] FUNGSI-FUNGSI PENGOLAHAN DATA & ANALITIK ---

@st.cache_data(ttl=30)
def get_data(endpoint):
    """Fungsi generik untuk mengambil data dari backend API."""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if 'entry_timestamp' in df.columns:
            df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'], errors='coerce').dt.tz_localize(None)
        if 'exit_timestamp' in df.columns:
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'], errors='coerce').dt.tz_localize(None)
        return df
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Koneksi API Gagal: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_advanced_metrics(_df):
    """Menghitung metrik statistik kuantitatif dari DataFrame trade."""
    if _df.empty or 'status' not in _df.columns or 'pnl_percent' not in _df.columns:
        return {"total_pnl_percent": 0, "total_trades": 0, "win_rate_percent": 0, "profit_factor": 0,
                "sharpe_ratio": 0, "max_drawdown_percent": 0, "expectancy_percent": 0,
                "winning_trades": 0, "losing_trades": 0}

    df = _df[_df['status'] == 'closed'].dropna(subset=['pnl_percent'])
    if df.empty:
        return {"total_pnl_percent": 0, "total_trades": 0, "win_rate_percent": 0, "profit_factor": 0,
                "sharpe_ratio": 0, "max_drawdown_percent": 0, "expectancy_percent": 0,
                "winning_trades": 0, "losing_trades": 0}

    total_trades = len(df); pnl = df['pnl_percent']
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    total_profit = wins.sum(); total_loss = abs(losses.sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_return = pnl.mean(); std_return = pnl.std()
    sharpe_ratio = (avg_return / std_return) * np.sqrt(365) if std_return > 0 else 0
    
    df_sorted = df.sort_values(by='exit_timestamp'); cumulative_pnl = df_sorted['pnl_percent'].cumsum()
    running_max = cumulative_pnl.cummax(); drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max() if not drawdown.empty else 0
    
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0
    expectancy = ((win_rate / 100) * avg_win) - ((1 - win_rate / 100) * avg_loss)
    
    return {"total_pnl_percent": total_profit - total_loss, "total_trades": total_trades,
            "winning_trades": len(wins), "losing_trades": len(losses), "win_rate_percent": win_rate,
            "profit_factor": profit_factor, "sharpe_ratio": sharpe_ratio,
            "max_drawdown_percent": max_drawdown, "expectancy_percent": expectancy}
            
@st.cache_data
def generate_pdf_report(_df, _metrics, date_range):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Apex Trading Bot - Performance Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Periode: {date_range[0].strftime('%Y-%m-%d')} hingga {date_range[1].strftime('%Y-%m-%d')}", 0, 1, 'C')
    pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Key Performance Indicators", 0, 1)
    pdf.set_font("Arial", '', 10)
    metrics_to_show = {"Total Net P/L (%)": f"{_metrics['total_pnl_percent']:.2f}%", "Win Rate (%)": f"{_metrics['win_rate_percent']:.2f}%",
                       "Profit Factor": f"{_metrics['profit_factor']:.2f}", "Sharpe Ratio (Annualized)": f"{_metrics['sharpe_ratio']:.2f}",
                       "Maximum Drawdown (%)": f"{_metrics['max_drawdown_percent']:.2f}%", "Total Trades": str(_metrics['total_trades'])}
    for key, value in metrics_to_show.items():
        pdf.cell(95, 7, f"  {key}:", 'B', 0); pdf.cell(95, 7, value, 'B', 1, 'R')
    pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Top 5 & Bottom 5 Trades", 0, 1)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(40, 6, 'Pair', 1, 0, 'C'); pdf.cell(40, 6, 'Strategy', 1, 0, 'C');
    pdf.cell(40, 6, 'Entry Date', 1, 0, 'C'); pdf.cell(30, 6, 'P/L (%)', 1, 1, 'C')
    pdf.set_font("Arial", '', 8)
    closed_trades = _df[_df['status'] == 'closed'].sort_values('pnl_percent', ascending=False).dropna(subset=['entry_timestamp', 'pnl_percent'])
    for _, row in pd.concat([closed_trades.head(5), closed_trades.tail(5)]).iterrows():
        pdf.cell(40, 5, str(row['pair']), 1); pdf.cell(40, 5, str(row['strategy_type']), 1)
        pdf.cell(40, 5, row['entry_timestamp'].strftime('%Y-%m-%d'), 1); pdf.cell(30, 5, f"{row['pnl_percent']:.2f}%", 1, 1, 'R')
    pdf.ln(5); pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "This report is generated for analytical purposes and does not constitute investment advice.", 0, 1, 'C')
    return pdf.output(dest='S').encode('latin-1')


# --- [4] TAMPILAN UTAMA DASHBOARD ---
st.title("Apex Trading Analytics Terminal")

signals_df = get_data("signals")
master_df = get_data("trades")

# --- Live Signal Radar & Analysis ---
st.header("📡 Live Signal Radar")
if not signals_df.empty:
    cols = st.columns([3, 2])
    with cols[0]:
        st.dataframe(signals_df.sort_values(by='timestamp', ascending=False), use_container_width=True)
    with cols[1]:
        signal_data = signals_df.iloc[0]
        st.subheader(f"Analisis Sinyal Terbaru: `{signal_data['pair'].upper()}`")
        kpi_cols = st.columns(2)
        kpi_cols[0].metric("Strategi", signal_data['strategy'].upper())
        kpi_cols[1].metric("ADX (M15)", f"{signal_data['adx']:.2f}")
        kpi_cols[0].metric("Harga vs SMA 200 (H4)", f"{signal_data.get('price_vs_sma_h4_percent', 0):.2f}%")
        status_color = "🟢 Aktif" if signal_data.get('golden_cross_m15_active', False) else "🔴 Tidak Aktif"
        kpi_cols[1].markdown(f"**Golden Cross (M15):** {status_color}")
else:
    st.info("Saat ini tidak ada sinyal trading aktif yang terdeteksi. Bot sedang memantau pasar...")
st.markdown("---")

# --- Expander untuk Filter & Simulasi ---
with st.expander("⚙️ Filter Analisis & Opsi Laporan", expanded=True):
    if not master_df.empty:
        # Fitur #4: Simulasi "What-If" (Placeholder)
        st.markdown("**Simulasi 'What-If' pada Data Historis:**")
        sim_min_adx = st.slider("Filter Trade Berdasarkan ADX Saat Masuk ≥", 20, 60, 25)
        st.caption("Info: Fitur simulasi memerlukan data 'ADX saat masuk' untuk disimpan pada setiap trade.")
        
        # Filter standar
        col1, col2, col3 = st.columns(3)
        with col1:
            date_range = st.date_input("Pilih Rentang Waktu", value=(master_df['entry_timestamp'].min().date(), datetime.now().date()))
        with col2:
            selected_pairs = st.multiselect("Pilih Aset", sorted(master_df['pair'].unique()), default=sorted(master_df['pair'].unique()))
        with col3:
            selected_strategies = st.multiselect("Pilih Strategi", sorted(master_df['strategy_type'].unique()), default=sorted(master_df['strategy_type'].unique()))

        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
        filtered_df = master_df[(master_df['entry_timestamp'] >= start_date) & (master_df['entry_timestamp'] <= end_date) &
                                (master_df['pair'].isin(selected_pairs)) & (master_df['strategy_type'].isin(selected_strategies))].copy()
    else:
        st.info("Filter dan simulasi akan aktif setelah ada data transaksi.")
        filtered_df = pd.DataFrame()

# --- Tampilan Utama (setelah expander) ---
st.header("Key Performance Indicators (KPIs)")
if not filtered_df.empty:
    metrics = calculate_advanced_metrics(filtered_df)
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

    if 'date_range' in locals():
        pdf_data = generate_pdf_report(filtered_df, metrics, date_range)
        st.sidebar.download_button(label="📄 Unduh Laporan PDF", data=pdf_data,
                                   file_name=f"apex_report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
else:
    st.info("Metrik akan ditampilkan di sini setelah ada data yang cocok dengan filter.")

st.markdown("---")
st.header("Visualisasi Performa")

# --- Heatmap Kalender ---
if not filtered_df.empty and 'exit_timestamp' in filtered_df.columns and not filtered_df['exit_timestamp'].isnull().all():
    closed_trades_cal = filtered_df[filtered_df['status'] == 'closed'].dropna(subset=['exit_timestamp', 'pnl_percent'])
    if not closed_trades_cal.empty:
        daily_pnl = closed_trades_cal.set_index('exit_timestamp')['pnl_percent'].resample('D').sum()
        try:
            fig, ax = calmap.calendarplot(daily_pnl, monthticks=3, daylabels='MTWTFSS', cmap='viridis', fillcolor='lightgrey', linewidth=0.5, fig_kws=dict(figsize=(12, 5)))
            st.pyplot(fig)
        except Exception as e:
            st.caption(f"Tidak dapat membuat heatmap: {e}")
    else:
        st.info("Menunggu data transaksi yang ditutup untuk menampilkan heatmap.")
else:
    st.info("Tidak ada data untuk menampilkan heatmap.")

# --- Tabs ---
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
