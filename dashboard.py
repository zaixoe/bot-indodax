# ==============================================================================
#           APEX QUANTUM ANALYTICS TERMINAL (AQAT) - v1.7 (UI/UX Refined)
# ==============================================================================
# SEBUAH DASHBOARD STREAMLIT UNTUK ANALISIS PERFORMA TRADING BOT TINGKAT LANJUT
# VERSI INI MEMINDAHKAN TOMBOL UNDUH PDF KE HALAMAN UTAMA UNTUK INTUITIVITAS
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

def get_live_signals(endpoint):
    """Fungsi untuk mengambil sinyal live (TIDAK di-cache)."""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except requests.exceptions.RequestException:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_trades_data(endpoint):
    """Fungsi untuk mengambil data historis trade."""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'entry_timestamp' in df.columns:
            df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'], errors='coerce').dt.tz_localize(None)
        if 'exit_timestamp' in df.columns:
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'], errors='coerce').dt.tz_localize(None)
        return df
    except requests.exceptions.RequestException as e:
        raise e # Lempar kembali error agar bisa ditangani di luar

@st.cache_data
def calculate_advanced_metrics(_df):
    """
    Versi 1.6: Membersihkan data P/L dengan lebih tangguh sebelum kalkulasi drawdown.
    """
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
    # --- [BLOK PERBAIKAN YANG SUDAH DIRAPIKAN] ---
    if 'exit_timestamp' in df.columns:
        # Hapus baris dengan exit_timestamp yang kosong untuk pengurutan yang andal
        df.dropna(subset=['exit_timestamp'], inplace=True)
        df.sort_values(by='exit_timestamp', inplace=True)
        
        # Buat kolom kumulatif HANYA dari data numerik yang valid
        df['cumulative_pnl'] = df['pnl_percent'].cumsum()
        
        # Hitung drawdown
        running_max = df['cumulative_pnl'].cummax()
        drawdown = running_max - df['cumulative_pnl']
        max_drawdown = drawdown.max() if not drawdown.empty else 0
    # --- [AKHIR BLOK PERBAIKAN] ---
    
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
    """
    Versi 1.5: Membuat laporan PDF dengan menghapus kolom objek (trade_data).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Apex Trading Bot - Performance Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        pdf.cell(0, 10, f"Periode: {date_range[0].strftime('%Y-%m-%d')} hingga {date_range[1].strftime('%Y-%m-%d')}", 0, 1, 'C')
    
    pdf.ln(5)
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
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top 5 & Bottom 5 Trades", 0, 1)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(40, 6, 'Pair', 1, 0, 'C'); pdf.cell(40, 6, 'Strategy', 1, 0, 'C');
    pdf.cell(40, 6, 'Entry Date', 1, 0, 'C'); pdf.cell(30, 6, 'P/L (%)', 1, 1, 'C')
    pdf.set_font("Arial", '', 8)

    closed_trades = _df[(_df['status'] == 'closed')].copy()
    closed_trades['pnl_percent'] = pd.to_numeric(closed_trades['pnl_percent'], errors='coerce')
    closed_trades.dropna(subset=['pnl_percent', 'entry_timestamp'], inplace=True)

    if not closed_trades.empty:
        if 'trade_data' in closed_trades.columns:
            closed_trades_for_display = closed_trades.drop(columns=['trade_data'])
        else:
            closed_trades_for_display = closed_trades

        closed_trades_for_display.sort_values('pnl_percent', ascending=False, inplace=True)
        trades_to_display = pd.concat([closed_trades_for_display.head(5), closed_trades_for_display.tail(5)]).drop_duplicates()
        
        for _, row in trades_to_display.iterrows():
            pdf.cell(40, 5, str(row.get('pair', 'N/A')), 1)
            pdf.cell(40, 5, str(row.get('strategy_type', 'N/A')), 1)
            pdf.cell(40, 5, row['entry_timestamp'].strftime('%Y-%m-%d'), 1)
            pdf.cell(30, 5, f"{row['pnl_percent']:.2f}%", 1, 1, 'R')
    else:
        pdf.cell(0, 5, "Tidak ada data trade yang ditutup untuk ditampilkan.", 1, 1, 'C')

    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "This report is generated for analytical purposes and does not constitute investment advice.", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- [4] TAMPILAN UTAMA DASHBOARD ---
st.title("Apex Trading Analytics Terminal")

try:
    signals_df = get_live_signals("signals") 
    master_df = get_trades_data("trades")
except requests.exceptions.RequestException as e:
    st.error(f"🚨 Gagal terhubung ke API Bot: {e}. Pastikan bot sedang berjalan dan BASE_URL sudah benar.")
    # Buat DataFrame kosong agar sisa aplikasi tidak crash
    signals_df = pd.DataFrame()
    master_df = pd.DataFrame()

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
        is_active = str(signal_data.get('golden_cross_m15_active', 'False')) == 'True'
        status_color = "🟢 Aktif" if is_active else "🔴 Tidak Aktif"
        kpi_cols[1].markdown(f"**Golden Cross (M15):** {status_color}")
else:
    st.info("Saat ini tidak ada sinyal trading aktif yang terdeteksi. Bot sedang memantau pasar...")
st.markdown("---")

with st.expander("⚙️ Filter Analisis & Opsi Laporan", expanded=True):
    if not master_df.empty and 'entry_timestamp' in master_df.columns:
        min_date = master_df['entry_timestamp'].min().date()
        max_date = datetime.now().date()
        date_range = st.date_input("Pilih Rentang Waktu", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        
        col2, col3 = st.columns(2)
        with col2:
            all_pairs = sorted(master_df['pair'].unique())
            selected_pairs = st.multiselect("Pilih Aset", all_pairs, default=all_pairs)
        with col3:
            all_strategies = sorted(master_df['strategy_type'].dropna().unique())
            selected_strategies = st.multiselect("Pilih Strategi", all_strategies, default=all_strategies)

        if len(date_range) == 2:
            start_date = datetime.combine(date_range[0], datetime.min.time())
            end_date = datetime.combine(date_range[1], datetime.max.time())
            
            filtered_df = master_df[
                (master_df['entry_timestamp'] >= start_date) & 
                (master_df['entry_timestamp'] <= end_date) &
                (master_df['pair'].isin(selected_pairs)) & 
                (master_df['strategy_type'].isin(selected_strategies))
            ].copy()
        else:
            filtered_df = pd.DataFrame()
    else:
        st.info("Filter akan aktif setelah ada data transaksi.")
        filtered_df = pd.DataFrame()

st.header("Key Performance Indicators (KPIs)")
if not filtered_df.empty:
    metrics = calculate_advanced_metrics(filtered_df)
    cols = st.columns(4)
    cols[0].metric("Total Net P/L (%)", f"{metrics['total_pnl_percent']:.2f}%")
    cols[1].metric("Win Rate", f"{metrics['win_rate_percent']:.1f}%")
    cols[2].metric("Profit Factor", f"{metrics['profit_factor']:.2f}" if not math.isinf(metrics['profit_factor']) else "∞")
    cols[3].metric("Sharpe Ratio (Ann.)", f"{metrics['sharpe_ratio']:.2f}")
    
    cols_b = st.columns(4)
    cols_b[0].metric("Maximum Drawdown (%)", f"{metrics['max_drawdown_percent']:.2f}%")
    cols_b[1].metric("Expectancy per Trade (%)", f"{metrics['expectancy_percent']:.2f}%")
    cols_b[2].metric("Total Trades", metrics['total_trades'])
    cols_b[3].metric("Menang / Kalah", f"{metrics['winning_trades']} / {metrics['losing_trades']}")

    # --- [PERUBAHAN LOKASI TOMBOL UNDUH] ---
    if 'date_range' in locals() and len(date_range) == 2:
        pdf_data = generate_pdf_report(filtered_df, metrics, date_range)
        # Tombol sekarang ada di sini, di bawah KPI, bukan di sidebar.
        st.download_button(
            label="📄 Unduh Laporan PDF",
            data=pdf_data,
            file_name=f"apex_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    # --- [AKHIR PERUBAHAN] ---

else:
    st.info("Metrik akan ditampilkan di sini setelah ada data yang cocok dengan filter.")

st.markdown("---")
st.header("Visualisasi Performa")

st.subheader("🗓️ Heatmap Kalender P/L Harian")
if not filtered_df.empty:
    df_for_cal = filtered_df.copy()
    df_for_cal['pnl_percent'] = pd.to_numeric(df_for_cal['pnl_percent'], errors='coerce')
    
    closed_trades_cal = df_for_cal[
        (df_for_cal['status'] == 'closed') & 
        (df_for_cal['pnl_percent'].notna()) &
        (df_for_cal['exit_timestamp'].notna())
    ]
    
    if not closed_trades_cal.empty:
        daily_pnl = closed_trades_cal.set_index('exit_timestamp')['pnl_percent'].resample('D').sum()
        start_cal_date = daily_pnl.index.min()
        end_cal_date = datetime.now()
        all_days = pd.date_range(start=start_cal_date, end=end_cal_date, freq='D')
        daily_pnl = daily_pnl.reindex(all_days, fill_value=0)
        dates = daily_pnl.index
        weekdays = dates.dayofweek
        weeks = dates.isocalendar().week
        hover_text = [f"{date.strftime('%Y-%m-%d')}<br>P/L: {pnl:.2f}%" for date, pnl in daily_pnl.items()]
        fig = go.Figure(data=go.Heatmap(
            z=daily_pnl.values, x=weeks, y=weekdays,
            text=hover_text, hoverinfo='text',
            colorscale='Viridis', showscale=False
        ))
        fig.update_layout(
            yaxis=dict(tickmode='array', tickvals=list(range(7)), ticktext=['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min']),
            xaxis_title="Minggu dalam Setahun", yaxis_title="", height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Menunggu data transaksi yang ditutup untuk menampilkan heatmap.")
else:
    st.info("Tidak ada data untuk menampilkan heatmap.")

tab1, tab2, tab3 = st.tabs(["📈 Kurva Ekuitas", "📊 Analisis per Aset", "📜 Detail Transaksi"])
with tab1:
    if not filtered_df.empty:
        equity_df = filtered_df[filtered_df['status'] == 'closed'].copy()
        
        # --- [PERBAIKAN KUNCI] ---
        # 1. Paksa konversi ke numerik, ubah semua yang gagal menjadi NaN.
        equity_df['pnl_percent'] = pd.to_numeric(equity_df['pnl_percent'], errors='coerce')
        
        # 2. Hapus semua baris di mana pnl_percent adalah NaN (tidak valid)
        equity_df.dropna(subset=['pnl_percent', 'exit_timestamp'], inplace=True)
        
        # 3. (Opsional tapi direkomendasikan) Ubah tipe data kolom menjadi float untuk konsistensi.
        equity_df['pnl_percent'] = equity_df['pnl_percent'].astype(float)
        # --- [AKHIR PERBAIKAN] ---
        if not equity_df.empty:
            equity_df.sort_values(by='exit_timestamp', inplace=True)
            equity_df['cumulative_pnl'] = equity_df['pnl_percent'].cumsum()
            equity_df['running_max'] = equity_df['cumulative_pnl'].cummax()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_df['exit_timestamp'], y=equity_df['running_max'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=equity_df['exit_timestamp'], y=equity_df['cumulative_pnl'], fill='tonexty', mode='lines', line_color='cyan', name='Equity Curve'))
            fig.update_layout(title_text='Kurva Pertumbuhan Ekuitas dengan Periode Drawdown (Area Abu-abu)', xaxis_title='Tanggal', yaxis_title='Total P/L Kumulatif (%)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Menunggu data transaksi yang ditutup untuk menampilkan Kurva Ekuitas.")
    else:
        st.info("Tidak ada data untuk menampilkan Kurva Ekuitas.")

with tab2:
    if not filtered_df.empty:
        pnl_by_pair_df = filtered_df[(filtered_df['status'] == 'closed')].copy()
        pnl_by_pair_df['pnl_percent'] = pd.to_numeric(pnl_by_pair_df['pnl_percent'], errors='coerce')
        pnl_by_pair_df.dropna(subset=['pnl_percent'], inplace=True)
        if not pnl_by_pair_df.empty:
            pnl_by_pair = pnl_by_pair_df.groupby('pair')['pnl_percent'].sum().sort_values(ascending=False)
            fig_pair_pnl = px.bar(pnl_by_pair, title="Total P/L (%) per Aset", labels={'value': 'Total P/L (%)', 'pair': 'Aset'})
            st.plotly_chart(fig_pair_pnl, use_container_width=True)
        else:
            st.info("Menunggu data transaksi yang ditutup untuk menampilkan Analisis per Aset.")
    else:
        st.info("Tidak ada data untuk menampilkan Analisis per Aset.")
        
with tab3:
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.info("Tidak ada data yang cocok dengan filter untuk menampilkan Detail Transaksi.")
