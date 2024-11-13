import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# サンプルデータ作成
dates = pd.date_range(start="2017-01-01", end="2017-01-31")
values = np.random.rand(len(dates))

# グラフ描画
fig, ax = plt.subplots()
ax.plot(dates, values, label="Anomaly Score")

# 日付範囲を指定して背景色を付ける
start_date = datetime(2017, 1, 10)
end_date = datetime(2017, 1, 20)
ax.axvspan(start_date, end_date, color='lightgray', alpha=0.5, label="Highlighted Period")

# その他の設定
ax.set_xlabel("Date")
ax.set_ylabel("Score")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# グラフを表示
plt.show()
