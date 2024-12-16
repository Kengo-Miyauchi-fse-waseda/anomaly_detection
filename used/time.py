import pandas as pd

# ISO 8601形式の時刻
time_str = "2017-12-29T20:40:00+00:00"

# datetime型に変換
time_obj = pd.to_datetime(time_str)
print(time_obj)
