import chardet

# ファイルのエンコーディングを推定
with open("data/haenkaze/1s_data/Analog_20240401_000000.csv", "rb") as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")