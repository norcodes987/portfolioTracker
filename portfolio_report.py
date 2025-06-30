import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import sys
import os
from dotenv import load_dotenv

load_dotenv()
log_file = os.path.join(os.path.dirname(__file__), "portfolio_log.txt")

# Redirect stdout and stderr to the file
sys.stdout = open(log_file, "a")
sys.stderr = sys.stdout

print("\n--- Script Run at", datetime.now(), "---\n")


def send_telegram_message(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    print("Telegram response status:", response.status_code)
    print("Telegram response body:", response.text)
    if not response.ok:
        print(f"Telegram error: {response.text}")

df = pd.read_csv(r"C:\Users\nor_t\OneDrive\Desktop\automation\stockPortfolio\Financials - Portfolio.csv")

# Filter rows with shares and avg price
has_shares = df['Shares'].notna() & df['Avg Price (SGD)'].notna()

# Aggregate by Symbol
agg_df = df[has_shares].groupby('Symbol', as_index=False).apply(
    lambda x: pd.Series({
        'Shares': x['Shares'].sum(),
        'Avg Price (SGD)': (x['Avg Price (SGD)'] * x['Shares']).sum() / x['Shares'].sum(),
        'Type.1': x['Type.1'].iloc[0]
    }),
    include_groups=False
).reset_index(drop=True)

# Fetch exchange rates dynamically

try:
    response = requests.get("https://open.er-api.com/v6/latest/USD")
    data = response.json()
    rates = data['rates']
    usd_to_sgd = rates['SGD']
    usd_to_hkd = rates['HKD']
    # Convert HKD to SGD: HKD->USD->SGD, so:
    hkd_to_sgd = usd_to_sgd / usd_to_hkd
    print(f"USD to SGD: {usd_to_sgd}")
    print(f"HKD to SGD: {hkd_to_sgd}")
except Exception as e:
    print("Error fetching exchange rates:", e)

# Filter only rows with shares
has_shares = df['Shares'].notna() & df['Avg Price (SGD)'].notna()

results = []
# Process rows with shares
for _, row in agg_df.iterrows():
    symbol = row['Symbol']
    shares = float(row['Shares'])
    avg_price = float(row['Avg Price (SGD)'])
    type1 = row['Type.1']

    try:
        ticker = yf.Ticker(symbol)
        price = ticker.info['regularMarketPrice']

        # Convert price to SGD
        if symbol.endswith(".L"):
            price_sgd = price * usd_to_sgd
        elif symbol.endswith(".HK"):
            price_sgd = price * hkd_to_sgd
        elif symbol.endswith(".SI"):
            price_sgd = price
        else:
            price_sgd = price * usd_to_sgd

        current_value = shares * price_sgd
        invested = shares * avg_price
        pnl = current_value - invested

        results.append({
            'Symbol': symbol,
            'Type': type1,
            'Current Price (SGD)': round(price_sgd, 2),
            'Shares': shares,
            'Invested (SGD)': round(invested, 2),
            'Current Value (SGD)': round(current_value, 2),
            'P/L (SGD)': round(pnl, 2)
        })

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# Include manual P/L entries (rows without shares but with P/L)
manual_pnl_rows = df[~has_shares & df['Profit/Loss'].notna()]

for _, row in manual_pnl_rows.iterrows():
    symbol = row['Symbol']
    type1 = row['Type.1']
    pnl = float(row['Profit/Loss'])

    results.append({
        'Symbol': symbol,
        'Type': type1,
        'Current Price (SGD)': None,
        'Shares': None,
        'Invested (SGD)': None,
        'Current Value (SGD)': None,
        'P/L (SGD)': round(pnl, 2)
    })

for r in results:
    invested_val = r.get('Invested (SGD)')
    if invested_val is None:
        # For manual entries, try to get 'Invested' from original df
        invested_rows = df[df['Symbol'] == r['Symbol']]
        if not invested_rows.empty and 'Invested' in invested_rows.columns:
            invested_val = invested_rows['Invested'].iloc[0]
    
    if invested_val and invested_val != 0:
        r['P/L (SGD)%'] = round(100 * r['P/L (SGD)'] / invested_val, 2)
    else:
        r['P/L (SGD)%'] = None

results_df = pd.DataFrame(results)

# Sort by P/L % descending
results_df.sort_values(by='P/L (SGD)%', ascending=False, inplace=True)

print(results_df)
print("\nTotal Profit/Loss (SGD):", round(results_df['P/L (SGD)'].sum(), 2))


# Group by Type, compute total and average performance
type_summary = results_df.groupby('Type').agg({
    'P/L (SGD)': 'sum',
    'P/L (SGD)%': 'mean'
}).rename(columns={
    'P/L (SGD)': 'Total P/L (SGD)',
    'P/L (SGD)%': 'Avg P/L (%)'
}).sort_values(by='Avg P/L (%)', ascending=False).reset_index()
# Add rank column
type_summary.insert(0, 'Rank', range(1, len(type_summary) + 1))

# Print ranked table
print("\nPerformance by Type (Ranked):")
print(type_summary.to_string(index=False))

# Function to convert a DataFrame to a text table
def df_to_telegram_table(df):
    df = df[[
        'Symbol', 'Current Price (SGD)', 'Shares',
        'Invested (SGD)', 'Current Value (SGD)',
        'P/L (SGD)', 'P/L (SGD)%'
    ]].fillna('').copy()

    # Format numbers
    df['Current Price (SGD)'] = df['Current Price (SGD)'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else '')
    df['Invested (SGD)'] = df['Invested (SGD)'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else '')
    df['Current Value (SGD)'] = df['Current Value (SGD)'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else '')
    df['P/L (SGD)'] = df['P/L (SGD)'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else '')
    df['P/L (SGD)%'] = df['P/L (SGD)%'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else '')

    headers = ['Symbol', 'Price', 'Shares', 'Invested', 'Current', 'P/L', 'P/L %']
    col_widths = [16, 8, 8, 10, 10, 10, 8]

    table = ' '.join(h.ljust(w) for h, w in zip(headers, col_widths)) + '\n'
    table += ' '.join('-' * w for w in col_widths) + '\n'

    for _, row in df.iterrows():
        # Determine gain/loss emoji
        try:
            pnl_val = float(row['P/L (SGD)'].replace(',', ''))
            emoji = '🟢' if pnl_val > 0 else '🔻' if pnl_val < 0 else '➖'
        except:
            emoji = ''

        row_items = [
            str(row['Symbol'])[:16],
            str(row['Current Price (SGD)'])[:8],
            str(row['Shares'])[:5],
            str(row['Invested (SGD)'])[:10],
            str(row['Current Value (SGD)'])[:10],
            str(row['P/L (SGD)'])[:10],
            str(row['P/L (SGD)%'])[:8],
        ]
        table += emoji + ' ' + ' '.join(item.ljust(w) for item, w in zip(row_items, col_widths)) + '\n'

    return table

def format_type_summary_table(df):
    # Define columns and their widths
    headers = ['Rank', 'Type', 'Total P/L (SGD)', 'Avg P/L (%)']
    col_widths = [6, 30, 18, 14]

    # Header line
    table = ' '.join(h.ljust(w) for h, w in zip(headers, col_widths)) + '\n'
    table += ' '.join('-' * w for w in col_widths) + '\n'

    # Row lines
    for _, row in df.iterrows():
        row_items = [
            str(row['Rank']),
            str(row['Type'])[:30],
            f"{row['Total P/L (SGD)']:.2f}",
            f"{row['Avg P/L (%)']:.2f}" if pd.notnull(row['Avg P/L (%)']) else 'N/A'
        ]
        table += ' '.join(item.ljust(w) for item, w in zip(row_items, col_widths)) + '\n'

    return table
type_table = format_type_summary_table(type_summary)

# Create message text
message = "*📊 Daily Portfolio Summary*\n\n"
message += "*💼 Holdings Overview:*\n"
message += "```\n" + df_to_telegram_table(results_df) + "\n```"
message += "*📈 Performance by Type (Ranked):*\n"
message += "```\n" + type_table + "\n```"

# Set your Telegram credentials
bot_token = os.getenv("BOT_TOKEN")
chat_id = os.getenv("CHAT_ID")

# Send to Telegram
send_telegram_message(message, bot_token, chat_id)