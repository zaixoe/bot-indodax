# ==============================================================================
#           APEX STRATEGIC COMMAND CENTER (ASCC) - v2.0
# ==============================================================================
# DASHBOARD STREAMLIT UNTUK MANAJEMEN & ANALISIS PERFORMA BOT SECARA HOLISTIK
# FITUR: MODE SIMULASI, ANALISIS TERSEGMENTASI, ROLLING METRICS, MONTE CARLO
# ==============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import random

# --- [1] KONFIGURASI HALAMAN & GAYA ---
st.set_page_config(layout="wide", page_title="Apex Strategic Command", page_icon="🚀")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- [2] KONFIGURASI KONEKSI API ---
BASE_URL = st.secrets.get("BASE_URL", "http://127.0.0.1:5000/api/v1/dashboard")
API_KEY = st.secrets.get("DASHBOARD_API_KEY", "your-fallback-key")
HEADERS = {"X-API-Key": API_KEY}

# --- [3] FUNGSI-FUNGSI PENGOLAHAN DATA & ANALITIK ---

@st.cache_data(ttl=60) # Cache lebih singkat untuk data yang lebih dinamis
def get_data_from_bot(endpoint):
    """Fungsi generik untuk mengambil data dari API bot."""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Konversi kolom waktu jika ada
        for col in ['entry_timestamp', 'exit_timestamp', 'timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
        
        # Pastikan kolom boolean ada
        for bool_col in ['is_simulation', 'is_hunter_mode']:
            if bool_col not in df.columns:
                df[bool_col] = False

        return df
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Gagal terhubung ke API Bot ({endpoint}): {e}. Pastikan bot berjalan dan URL sudah benar.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi error saat memproses data dari endpoint {endpoint}: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_advanced_metrics(_df):
    """[v2.0] Menghitung metrik performa tingkat lanjut, termasuk analisis streak."""
    if _df.empty: return {}

    df = _df[_df['status'] == 'closed'].copy()
    df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce').fillna(0)
    df = df[df['pnl_percent'] > -100.1]
    
    if df.empty: return {}

    total_trades = len(df)
    pnl = df['pnl_percent']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    
    # Analisis Streak (Rentetan)
    df['win'] = (df['pnl_percent'] > 0).astype(int)
    df['streak_group'] = (df['win'] != df['win'].shift()).cumsum()
    streaks = df.groupby('streak_group').size()
    win_streaks = df[df['win'] == 1].groupby('streak_group').size()
    loss_streaks = df[df['win'] == 0].groupby('streak_group').size()

    # Hitung metrik lainnya
    df.sort_values(by='exit_timestamp', inplace=True, na_position='first')
    df['cumulative_pnl'] = df['pnl_percent'].cumsum()
    running_max = df['cumulative_pnl'].cummax()
    drawdown = running_max - df['cumulative_pnl']
    
    total_profit = wins.sum()
    total_loss = abs(losses.sum())
    
    metrics = {
        "total_pnl_percent": pnl.sum(),
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_percent": (len(wins) / total_trades * 100) if total_trades > 0 else 0,
        "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
        "sharpe_ratio": (pnl.mean() / pnl.std()) * np.sqrt(365) if pnl.std() > 0 else 0,
        "max_drawdown_percent": drawdown.max() if not drawdown.empty else 0,
        "avg_win_percent": wins.mean() if not wins.empty else 0,
        "avg_loss_percent": abs(losses.mean()) if not losses.empty else 0,
        "longest_win_streak": win_streaks.max() if not win_streaks.empty else 0,
        "longest_loss_streak": loss_streaks.max() if not loss_streaks.empty else 0,
    }
    metrics["expectancy_percent"] = ((metrics['win_rate_percent'] / 100) * metrics['avg_win_percent']) - \
                                    ((1 - metrics['win_rate_percent'] / 100) * metrics['avg_loss_percent'])
    
    return metrics

def format_metric(value, suffix="", precision=2):
    """Helper untuk memformat metrik agar rapi."""
    if isinstance(value, (int, float)):
        if math.isinf(value): return "∞"
        return f"{value:,.{precision}f}{suffix}"
    return str(value)

# --- [4] TAMPILAN UTAMA DASHBOARD ---
st.title("🚀 Apex Strategic Command Center")

# [FITUR BARU] Saklar utama untuk mode
mode = st.radio(
    "Pilih Mode Analisis:",
    ('Trading Riil', 'Mode Simulasi'),
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# Ambil semua data sekali di awal
master_df = get_data_from_bot("trades")

# Filter data berdasarkan mode yang dipilih
is_simulation_mode = (mode == 'Mode Simulasi')
if not master_df.empty:
    df_mode_filtered = master_df[master_df['is_simulation'] == is_simulation_mode].copy()
else:
    df_mode_filtered = pd.DataFrame()


# Tampilkan data yang relevan dengan mode
if is_simulation_mode:
    st.info("Anda sedang melihat data dari **Mode Simulasi**. Transaksi ini tidak menggunakan uang riil.")
else:
    st.info("Anda sedang melihat data dari **Trading Riil**.")

# Filter lebih lanjut
with st.expander("⚙️ Filter Lanjutan & Opsi", expanded=False):
    if not df_mode_filtered.empty and 'entry_timestamp' in df_mode_filtered.columns:
        min_date = df_mode_filtered['entry_timestamp'].min().date()
        max_date = datetime.now().date()
        date_range = st.date_input("Pilih Rentang Waktu", value=(min_date, max_date), min_value=min_date, max_date=max_date)
        
        col1, col2 = st.columns(2)
        all_pairs = sorted(df_mode_filtered['pair'].unique())
        selected_pairs = col1.multiselect("Pilih Aset", all_pairs, default=all_pairs)
        
        all_strategies = sorted(df_mode_filtered['strategy_type'].dropna().unique())
        selected_strategies = col2.multiselect("Pilih Strategi", all_strategies, default=all_strategies)

        if len(date_range) == 2:
            start_date = datetime.combine(date_range[0], datetime.min.time())
            end_date = datetime.combine(date_range[1], datetime.max.time())
            
            filtered_df = df_mode_filtered[
                (df_mode_filtered['entry_timestamp'].dt.date >= date_range[0]) & 
                (df_mode_filtered['entry_timestamp'].dt.date <= date_range[1]) &
                (df_mode_filtered['pair'].isin(selected_pairs)) & 
                (df_mode_filtered['strategy_type'].isin(selected_strategies))
            ].copy()
        else:
            filtered_df = pd.DataFrame()
    else:
        st.write("Filter akan aktif setelah ada data transaksi.")
        filtered_df = pd.DataFrame()

# Hitung metrik dari data yang sudah difilter
metrics = calculate_advanced_metrics(filtered_df)

# --- TABS UTAMA ---
tab_kpi, tab_perf, tab_sim, tab_data, tab_signals = st.tabs([
    "📊 Key Performance Indicators",
    "📈 Analisis Performa",
    "🔮 Simulasi Monte Carlo",
    "📜 Detail Transaksi",
    "📡 Radar Sinyal Live"
])

with tab_kpi:
    st.header("Metrik Kinerja Utama")
    if metrics:
        cols = st.columns(4)
        cols[0].metric("Total Net P/L (%)", format_metric(metrics.get('total_pnl_percent'), suffix='%'))
        cols[1].metric("Win Rate", format_metric(metrics.get('win_rate_percent'), suffix='%', precision=1))
        cols[2].metric("Profit Factor", format_metric(metrics.get('profit_factor')))
        cols[3].metric("Total Trades", metrics.get('total_trades', 0))
        
        st.markdown("---")
        
        cols_b = st.columns(4)
        cols_b[0].metric("Maximum Drawdown (%)", format_metric(metrics.get('max_drawdown_percent'), suffix='%'))
        cols_b[1].metric("Expectancy per Trade (%)", format_metric(metrics.get('expectancy_percent'), suffix='%'))
        cols_b[2].metric("Avg Win / Avg Loss (%)", f"{format_metric(metrics.get('avg_win_percent'))} / {format_metric(metrics.get('avg_loss_percent'))}")
        cols_b[3].metric("Longest Win / Loss Streak", f"{metrics.get('longest_win_streak', 0)} / {metrics.get('longest_loss_streak', 0)}")
    else:
        st.info("Metrik akan ditampilkan setelah ada data yang cocok dengan filter.")

with tab_perf:
    st.header("Visualisasi Performa Lanjutan")
    
    # [FITUR BARU] Analisis Performa Tersegmentasi
    st.subheader("🔬 Analisis Performa Tersegmentasi (Sankey Diagram)")
    if not filtered_df.empty:
        sankey_df = filtered_df[filtered_df['status'] == 'closed'].copy()
        sankey_df['mode'] = np.where(sankey_df['is_hunter_mode'], 'Hunter', 'Normal')
        
        if not sankey_df.empty:
            all_nodes = ['Profit', 'Loss'] + sankey_df['mode'].unique().tolist() + sankey_df['strategy_type'].unique().tolist()
            node_map = {name: i for i, name in enumerate(all_nodes)}

            links = {'source': [], 'target': [], 'value': [], 'color': []}

            for mode_name, group in sankey_df.groupby('mode'):
                total_mode_pnl = group['pnl_percent'].sum()
                target_node = 'Profit' if total_mode_pnl > 0 else 'Loss'
                links['source'].append(node_map[mode_name])
                links['target'].append(node_map[target_node])
                links['value'].append(abs(total_mode_pnl))
                links['color'].append('rgba(50, 205, 50, 0.4)' if total_mode_pnl > 0 else 'rgba(255, 69, 0, 0.4)')

            for (mode_name, strategy_name), group in sankey_df.groupby(['mode', 'strategy_type']):
                total_strategy_pnl = group['pnl_percent'].sum()
                if abs(total_strategy_pnl) > 0.01:
                    links['source'].append(node_map[strategy_name])
                    links['target'].append(node_map[mode_name])
                    links['value'].append(abs(total_strategy_pnl))
                    links['color'].append('rgba(100, 149, 237, 0.4)')

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes),
                link=links)])
            fig.update_layout(title_text="Aliran Profit/Loss dari Strategi ke Mode", font_size=10, height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Analisis tersegmentasi memerlukan data transaksi.")

    # [FITUR BARU] Rolling Performance
    st.subheader("📈 Kinerja Bergulir (Rolling Win Rate)")
    if not filtered_df.empty:
        rolling_df = filtered_df[filtered_df['status'] == 'closed'].copy()
        rolling_df.sort_values(by='exit_timestamp', inplace=True)
        if len(rolling_df) > 20:
            rolling_df['win'] = (rolling_df['pnl_percent'] > 0).astype(int)
            rolling_df['rolling_win_rate'] = rolling_df['win'].rolling(window=20).mean() * 100
            
            fig_rolling = px.line(rolling_df, x='exit_timestamp', y='rolling_win_rate', 
                                  title="Win Rate Bergulir (Jendela 20 Trade)",
                                  labels={'exit_timestamp': 'Tanggal', 'rolling_win_rate': 'Win Rate (%)'})
            fig_rolling.add_hline(y=50, line_dash="dash", line_color="red")
            st.plotly_chart(fig_rolling, use_container_width=True)
    else:
        st.info("Grafik kinerja bergulir memerlukan setidaknya 20 trade.")

with tab_sim:
    st.header("🔮 Proyeksi Masa Depan dengan Simulasi Monte Carlo")
    if not filtered_df.empty:
        pnl_history = filtered_df[filtered_df['status'] == 'closed']['pnl_percent'].dropna().tolist()
        if len(pnl_history) > 10:
            sim_cols = st.columns(3)
            num_simulations = sim_cols[0].number_input("Jumlah Simulasi", 10, 1000, 100)
            num_trades_future = sim_cols[1].number_input("Jumlah Trade ke Depan", 10, 500, 100)
            
            if st.button("Jalankan Simulasi Monte Carlo"):
                with st.spinner("Menjalankan simulasi... Ini mungkin memakan waktu beberapa detik."):
                    simulation_results = []
                    all_paths = []
                    for _ in range(num_simulations):
                        equity_path = [100] # Mulai dengan ekuitas 100
                        for _ in range(num_trades_future):
                            random_pnl = random.choice(pnl_history) / 100
                            equity_path.append(equity_path[-1] * (1 + random_pnl))
                        simulation_results.append(equity_path[-1])
                        all_paths.append(equity_path)

                    fig = go.Figure()
                    for path in all_paths:
                        fig.add_trace(go.Scatter(x=list(range(len(path))), y=path, mode='lines', line=dict(width=1, color='rgba(173, 216, 230, 0.3)')))
                    
                    fig.update_layout(title=f"{num_simulations} Proyeksi Jalur Ekuitas untuk {num_trades_future} Trade ke Depan",
                                      xaxis_title="Jumlah Trade", yaxis_title="Pertumbuhan Ekuitas (Mulai dari 100)", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Tampilkan statistik hasil simulasi
                    final_equities = np.array(simulation_results)
                    prob_profit = (np.sum(final_equities > 100) / len(final_equities)) * 100
                    
                    st.subheader("Hasil Statistik Simulasi")
                    res_cols = st.columns(4)
                    res_cols[0].metric("Rata-rata Hasil Akhir", f"{final_equities.mean():.2f}")
                    res_cols[1].metric("Hasil Terbaik", f"{final_equities.max():.2f}")
                    res_cols[2].metric("Hasil Terburuk", f"{final_equities.min():.2f}")
                    res_cols[3].metric("Probabilitas Profit", f"{prob_profit:.1f}%")

with tab_data:
    st.header("📜 Detail Transaksi")
    if not filtered_df.empty:
        # [FITUR BARU] Kode Warna P/L
        def color_pnl(val):
            if pd.isna(val): return ''
            color = '#3D9970' if val > 0 else '#FF4136' if val < 0 else ''
            return f'color: {color}'

        st.dataframe(
            filtered_df.style.applymap(color_pnl, subset=['pnl_percent']),
            use_container_width=True
        )
    else:
        st.info("Tidak ada data transaksi yang cocok dengan filter.")

with tab_signals:
    st.header("📡 Radar Sinyal Live")
    signals_df = get_data_from_bot("signals")
    if not signals_df.empty:
        st.write("Daftar kandidat trade teratas yang sedang dipertimbangkan oleh bot (diurutkan berdasarkan skor).")
        st.dataframe(
            signals_df.sort_values(by='score', ascending=False),
            use_container_width=True
        )
    else:
        st.info("Saat ini tidak ada sinyal trading aktif yang terdeteksi. Bot sedang memantau pasar...")
