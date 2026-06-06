import pandas as pd
import re


file_path = r"Dataset\raw\ALL.TXT" 

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()


df = pd.DataFrame(lines, columns=["raw"])


df["raw"] = df["raw"].str.strip()


print(df.head())
print("\nToplam satır:", len(df))


print("Boş satır:", df["raw"].isna().sum())
print("Empty string:", (df["raw"] == "").sum())


## PARSING ##

def parse_line(line):
    parts = line.split()

    
    if len(parts) < 8:
        return [None]*6

    timestamp = parts[0]
    freq = parts[1]
    snr = parts[4]
    dt = parts[5]
    audio_freq = parts[6]

    
    message = " ".join(parts[7:])

    return [timestamp, freq, snr, dt, audio_freq, message]



parsed = df["raw"].apply(parse_line)


df_parsed = pd.DataFrame(parsed.tolist(), columns=[
    "timestamp", "freq", "snr", "dt", "audio_freq", "message"
])

print(df_parsed.head())


##GRID extraction ##

def extract_tx_and_grid(msg):

    if not isinstance(msg, str):
        return None, None

    parts = msg.split()

    if len(parts) < 3:
        return None, None

    grid_match = re.search(r'\b[A-R]{2}[0-9]{2}\b', msg)
    if not grid_match:
        return None, None

    grid = grid_match.group(0)

    
    if parts[-1] != grid:
        return None, None

    callsign_pattern = r'^[A-Z0-9/]{3,}$'

    # CQ
    if parts[0] == "CQ":
        tx = parts[1]

    # QSO
    else:
        tx = parts[1]

    
    if not re.match(callsign_pattern, tx):
        return None, None

    return tx, grid
    

df_parsed[["tx_callsign", "grid"]] = df_parsed["message"].apply(
    lambda x: pd.Series(extract_tx_and_grid(x))
)

print(df_parsed.head())


df_clean = df_parsed.dropna(subset=["grid", "tx_callsign"]).copy()

print("Kalan satır:", len(df_clean))


print("Toplam veri:", len(df_parsed))
print("Temiz veri:", len(df_clean))
print("Kayıp oranı:", 1 - len(df_clean)/len(df_parsed))


df_clean.to_csv("clean_ft8.csv", index=False)