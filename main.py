# Import library standar
import logging
import asyncio
import requests
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import datetime
from replit import db

# Import library eksternal
from telegram.ext import Application, CommandHandler
from telegram.error import TelegramError
from flask import Flask
from threading import Thread

# ==============================================================================
# BAGIAN KODE UNTUK SERVER WEB "KEEP-ALIVE" (AGAR GRATIS 24/7)
# ==============================================================================
app = Flask('')
@app.route('/')
def home():
    return "Bot is alive and running."
def run_web_server():
  app.run(host='0.0.0.0', port=8080)
def start_web_server_thread():
  web_thread = Thread(target=run_web_server)
  web_thread.start()
# ==============================================================================

try:
    import config
    TOKEN_BOT = config.BOT_TOKEN
    PRIMARY_USER_CHAT_ID = config.CHAT_ID
except ImportError:
    # Berhenti jika file config.py tidak ditemukan
    print("="*50)
    print("KESALAHAN: File 'config.py' tidak ditemukan.")
    print("Silakan buat file config.py dan isi dengan token Anda.")
    print("="*50)
    exit()

# --- PENGATURAN STRATEGI & ANALISIS ---
SIGNAL_INTERVAL_MENIT = 15
VERY_LOW_PROXIMITY = 1.0
LOW_PROXIMITY = 2.5
STRONG_MOMENTUM_CHANGE = 0.75
WEAK_MOMENTUM_CHANGE = 0.1
BUY_SCORE_THRESHOLD = 3
SELL_SCORE_THRESHOLD = -3
STOP_LOSS_PERCENT = 5
TAKE_PROFIT_PERCENT = 10

# --- Konfigurasi Logging ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Variabel Global ---
monitoring_active = False
monitor_task = None
coin_states = {}

async def send_telegram_message(context, chat_id, message):
    try:
        await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
    except TelegramError as e:
        logger.error(f"Gagal mengirim pesan ke {chat_id}: {e}")

def get_full_market_summary():
    url = "https://indodax.com/api/summaries"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Koneksi ke API Indodax gagal: {e}")
        return None

def get_user_timezone(user_id):
    user_id_str = str(user_id)
    return db.get(user_id_str, "UTC")

def get_top_10_status_message(user_id):
    full_summary = get_full_market_summary()
    if not full_summary: return "<i>Gagal mengambil data pasar dari Indodax saat ini.</i>"
    try:
        tickers = full_summary.get('tickers', {})
        idr_pairs = {k: v for k, v in tickers.items() if k.endswith('_idr')}
        if not idr_pairs: return "<i>Tidak ada pasangan koin IDR yang ditemukan.</i>"
        sorted_pairs = sorted(idr_pairs.items(), key=lambda item: float(item[1]['vol_idr']), reverse=True)

        message_lines = [f"📊 <b>Harga 10 Koin Teratas</b> (Volume 24 Jam)"]
        message_lines.append("----------------------------------")
        for pair_name, pair_data in sorted_pairs[:10]:
            last_price = float(pair_data.get('last', 0))
            message_lines.append(f"🔸 <b>{pair_name.replace('_idr', '').upper()}</b>: Rp {last_price:,.0f}")

        user_tz_str = get_user_timezone(user_id)
        user_tz = ZoneInfo(user_tz_str)
        first_coin_data = sorted_pairs[0][1]
        server_timestamp = int(first_coin_data.get('server_time', 0))
        naive_utc_time = datetime.fromtimestamp(server_timestamp)
        aware_utc_time = naive_utc_time.replace(tzinfo=ZoneInfo("UTC"))
        local_time = aware_utc_time.astimezone(user_tz)
        message_lines.append(f"\n<i>Update pada: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({user_tz_str})</i>")
        return "\n".join(message_lines)
    except Exception as e:
        logger.error(f"Gagal memformat pesan status: {e}")
        return "<i>Terjadi error saat memproses data pasar.</i>"

def calculate_signal_score(pair_data, previous_price):
    last_price = float(pair_data.get('last', 0))
    high_24h = float(pair_data.get('high', 0))
    low_24h = float(pair_data.get('low', 0))
    value_score = 0
    if last_price <= low_24h * (1 + VERY_LOW_PROXIMITY / 100): value_score = 2
    elif last_price <= low_24h * (1 + LOW_PROXIMITY / 100): value_score = 1
    elif last_price >= high_24h * (1 - VERY_LOW_PROXIMITY / 100): value_score = -2
    elif last_price >= high_24h * (1 - LOW_PROXIMITY / 100): value_score = -1
    momentum_score = 0
    if previous_price > 0:
        price_change_percent = ((last_price - previous_price) / previous_price) * 100
        if price_change_percent >= STRONG_MOMENTUM_CHANGE: momentum_score = 2
        elif price_change_percent >= WEAK_MOMENTUM_CHANGE: momentum_score = 1
        elif price_change_percent <= -STRONG_MOMENTUM_CHANGE: momentum_score = -2
        elif price_change_percent <= -WEAK_MOMENTUM_CHANGE: momentum_score = -1
    return value_score + momentum_score, value_score, momentum_score

async def market_analysis_loop(context):
    global coin_states
    while monitoring_active:
        logger.info("Memulai siklus analisis pasar kuantitatif...")
        signal_found_in_cycle = False
        full_summary = get_full_market_summary()
        if not full_summary or 'tickers' not in full_summary:
            await asyncio.sleep(SIGNAL_INTERVAL_MENIT * 60)
            continue

        tickers = full_summary.get('tickers', {})
        idr_pairs = {k: v for k, v in tickers.items() if k.endswith('_idr')}
        sorted_pairs = sorted(idr_pairs.items(), key=lambda item: float(item[1]['vol_idr']), reverse=True)
        top_10_pairs = [pair[0] for pair in sorted_pairs[:10]]

        for pair_name in top_10_pairs:
            try:
                pair_data = tickers.get(pair_name)
                last_price = float(pair_data.get('last', 0))
                if pair_name not in coin_states:
                    coin_states[pair_name] = {'previous_price': last_price, 'last_signal_score': 0}
                    continue
                previous_price = coin_states[pair_name]['previous_price']
                last_signal_score = coin_states[pair_name]['last_signal_score']
                total_score, value_score, momentum_score = calculate_signal_score(pair_data, previous_price)

                if total_score >= BUY_SCORE_THRESHOLD and last_signal_score < BUY_SCORE_THRESHOLD:
                    signal_found_in_cycle = True
                    coin_states[pair_name]['last_signal_score'] = total_score
                    stop_loss_price = last_price * (1 - STOP_LOSS_PERCENT / 100)
                    take_profit_price = last_price * (1 + TAKE_PROFIT_PERCENT / 100)
                    buy_message = (
                        f"🔥 <b>SINYAL BELI KUAT: {pair_name.upper()}</b> 🔥\n\n"
                        f"<b>Skor Analisis: {total_score}</b> (Nilai: {value_score}, Momentum: {momentum_score})\n"
                        f"<b>Harga:</b> Rp {last_price:,.0f}\n\n"
                        f"💡 <b><u>Saran Trading</u></b> 💡\n"
                        f"<b>Stop-Loss:</b> ~Rp {stop_loss_price:,.0f}\n"
                        f"<b>Take-Profit:</b> ~Rp {take_profit_price:,.0f}"
                    )
                    await send_telegram_message(context, PRIMARY_USER_CHAT_ID, buy_message)

                elif total_score <= SELL_SCORE_THRESHOLD and last_signal_score > SELL_SCORE_THRESHOLD:
                    signal_found_in_cycle = True
                    coin_states[pair_name]['last_signal_score'] = total_score
                    sell_message = (
                        f"❄️ <b>SINYAL JUAL KUAT: {pair_name.upper()}</b> ❄️\n\n"
                        f"<b>Skor Analisis: {total_score}</b> (Nilai: {value_score}, Momentum: {momentum_score})\n"
                        f"<b>Harga:</b> Rp {last_price:,.0f}"
                    )
                    await send_telegram_message(context, PRIMARY_USER_CHAT_ID, sell_message)

                elif SELL_SCORE_THRESHOLD < total_score < BUY_SCORE_THRESHOLD:
                     coin_states[pair_name]['last_signal_score'] = 0

                coin_states[pair_name]['previous_price'] = last_price

            except Exception as e:
                logger.error(f"Gagal menganalisis koin {pair_name}: {e}")
                continue

        if not signal_found_in_cycle:
            user_tz_str = get_user_timezone(PRIMARY_USER_CHAT_ID)
            user_tz = ZoneInfo(user_tz_str)
            now_local = datetime.now(user_tz)
            heartbeat_message = f"✅ <i>Analisis pasar selesai pukul {now_local.strftime('%H:%M:%S')} ({user_tz_str}). Tidak ada sinyal kuat. Memindai kembali dalam {SIGNAL_INTERVAL_MENIT} menit.</i>"
            await send_telegram_message(context, PRIMARY_USER_CHAT_ID, heartbeat_message)
            logger.info("Siklus selesai, tidak ada sinyal. Pesan Heartbeat terkirim.")

        await asyncio.sleep(SIGNAL_INTERVAL_MENIT * 60)

async def start_command(update, context):
    await update.message.reply_text(
        "👋 <b>Bot Analisis Kuantitatif Aktif!</b>\n\n"
        "Gunakan /set_timezone untuk mengatur zona waktu Anda (contoh: <code>/set_timezone Asia/Jakarta</code>).\n\n"
        "Perintah yang tersedia:\n"
        "▶️ /monitor - Memulai pemantauan sinyal.\n"
        "📊 /harga_10_koin_teratas - Melihat harga Top 10.\n"
        "⏹️ /stop - Menghentikan pemantauan.",
        parse_mode='HTML'
    )

async def monitor_command(update, context):
    global monitoring_active, monitor_task
    if not monitoring_active:
        monitoring_active = True
        await update.message.reply_text("✅ <b>Analisis kuantitatif dimulai!</b> Saya akan memindai pasar setiap 15 menit.", parse_mode='HTML')
        monitor_task = asyncio.create_task(market_analysis_loop(context))
    else:
        await update.message.reply_text("ℹ️ Pemantauan sudah aktif.", parse_mode='HTML')

async def harga_10_koin_teratas_command(update, context):
    user_id = update.effective_user.id
    await update.message.reply_text("<i>Mengambil data harga terkini...</i>", parse_mode='HTML')
    status_message = get_top_10_status_message(user_id)
    await send_telegram_message(context, user_id, status_message)

async def set_timezone_command(update, context):
    user_id = str(update.effective_user.id)
    if not context.args:
        await update.message.reply_text(
            "Gunakan format: <code>/set_timezone [Nama_Zona_Waktu]</code>\n\n"
            "Contoh:\n"
            "• <code>/set_timezone Asia/Jakarta</code> (WIB)\n"
            "Zona waktu Anda saat ini: <b>{}</b>".format(db.get(user_id, "UTC")),
            parse_mode='HTML'
        )
        return
    try:
        timezone_str = context.args[0]
        ZoneInfo(timezone_str)
        db[user_id] = timezone_str
        await update.message.reply_text(f"✅ Timezone Anda telah diatur ke: <b>{timezone_str}</b>", parse_mode='HTML')
    except ZoneInfoNotFoundError:
        await update.message.reply_text("❌ Zona waktu tidak valid. Gunakan format standar seperti 'Asia/Jakarta'.", parse_mode='HTML')

async def stop_command(update, context):
    global monitoring_active, monitor_task
    if monitoring_active:
        monitoring_active = False
        if monitor_task: monitor_task.cancel()
        monitor_task = None
        coin_states.clear()
        await update.message.reply_text("⏹️ <b>Analisis kuantitatif dihentikan.</b> State telah direset.", parse_mode='HTML')
    else:
        await update.message.reply_text("ℹ️ Pemantauan tidak aktif.", parse_mode='HTML')

def main():
    print(">>> MEMULAI BOT ANALISIS KUANTITATIF (PRODUCTION-GRADE) <<<")

    # Memulai server web "keep-alive" di thread terpisah
    start_web_server_thread()

    application = Application.builder().token(TOKEN_BOT).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("monitor", monitor_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("harga_10_koin_teratas", harga_10_koin_teratas_command))
    application.add_handler(CommandHandler("set_timezone", set_timezone_command))

    application.run_polling()

if __name__ == "__main__":
    main()
