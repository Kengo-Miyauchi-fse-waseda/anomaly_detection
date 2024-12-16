from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime

def visualize_events(event_file):
    event_dates=pd.read_csv(event_file)
    if (event_dates.shape[1]==1):type="line"
    else: type="range"
    if(type=="line"):
        events = event_dates["event_date"]
        for i in range(len(events)):
            event = pd.Timestamp(events[i])
            plt.axvline(event, color='orange', alpha=1.0)
    elif(type=="range"):
        start_date=event_dates['start_date']
        end_date=event_dates['end_date']
        # イベント区間の可視化
        for i in range(len(start_date)):
            start = datetime.strptime(start_date[i], "%Y-%m-%d")
            end = datetime.strptime(end_date[i], "%Y-%m-%d")
            plt.axvspan(start, end, color='orange', alpha=0.2)

def color_train_range(train_range):
    plt.axvspan(train_range[0], train_range[1], color='green', alpha=0.2)

def plot_by_date(log_plot, anomaly_score, timestamp, train_range, threshold, img_path, event_file=None):
    df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": anomaly_score})
    # DATATIMEを日付部分だけに変換（Y軸を同じにするため）
    if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
        df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
    df['Date'] = df['DATETIME'].dt.date

    # 散布図をプロット
    plt.figure(figsize=(40, 8))
    if log_plot:
        plt.yscale('log')
    for date, group in df.groupby('Date'):
        plt.scatter([date]*len(group), group['AnomalyScore'], alpha=0.8, marker='o', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Scores by Date")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.axhline(y=threshold, color='r', linestyle='solid')
    
    # 学習期間/イベント情報の可視化
    color_train_range(train_range)
    if(event_file!=None):visualize_events(event_file)
    
    # 凡例の表示を防止するための設定（重複する日付を削除）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(img_path)
    plt.clf()
    plt.close()