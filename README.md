# 🤖 Trading Alert Bots (Crypto & US Stocks)

This repository contains **two alert-based trading bots** designed for market analysis and decision support:

- **Crypto Bot** (`crypto_bot.py`)  
  Analyzes cryptocurrency spot markets using technical indicators and sends entry/exit alerts.

- **US Stock Bot** (`us_stock_bot.py`)  
  Analyzes US stock market data (NYSE / NASDAQ) and sends trading alerts.

> ⚠️ Both bots are **alert / paper-trading only**  
> ❌ No real orders are placed (no auto-trading)

---

## 📁 Project Structure

bot/
├── crypto_bot.py # Crypto trading alert bot
├── us_stock_bot.py # US stock trading alert bot
├── requirements.txt # Python dependencies
├── README.md
├── .gitignore
├── logs/ # Runtime logs (git ignored)
└── .env # Environment variables (not committed)

