from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import csv
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score

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

def get_date_range(start,end):
    dates = []
    current_date = start
    while current_date <= end:
        date = current_date.date()
        dates.append(date)
        current_date += timedelta(days=1)
    return dates

def get_event_range(event_files, before):
    event_dates = set()
    for event_file in event_files:
        events=pd.read_csv(event_file)
        if (events.shape[1]==1):type="line"
        else: type="range"
        if(type=="line"):
            dates = events["event_date"].values
            for date in dates:
                start = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=before)
                end = datetime.strptime(date, "%Y-%m-%d")
                event_dates.update(get_date_range(start,end))
        elif(type=="range"):
            start_date=events['start_date'].values
            end_date=events['end_date'].values
            for i in range(len(start_date)):
                start = datetime.strptime(start_date[i], "%Y-%m-%d") - timedelta(days=before)
                end = datetime.strptime(end_date[i], "%Y-%m-%d")
                event_dates.update(get_date_range(start,end))
    return list(event_dates)

def get_anomaly_dates(event_files, before):
    anomaly_dates = set()
    for event_file in event_files:
        events=pd.read_csv(event_file)
        if (events.shape[1]==1):type="line"
        else: type="range"
        #import pdb; pdb.set_trace()
        if(type=="line"):
            event_dates = events["event_date"].values
            for date in event_dates:
                anomaly_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=before)
                anomaly_dates.add(anomaly_date.date())
                #import pdb; pdb.set_trace()
        elif(type=="range"):
            start_date=events['start_date'].values
            end_date=events['end_date'].values
            for i in range(len(start_date)):
                start = datetime.strptime(start_date[i], "%Y-%m-%d") - timedelta(days=before)
                end = datetime.strptime(end_date[i], "%Y-%m-%d")
                anomaly_dates.update(get_date_range(start,end))
                #import pdb; pdb.set_trace()
    return list(anomaly_dates)

def get_normal_dates(event_files, before, range_days, anomaly_dates):
    """anomaly_dates に含まれない範囲の日付を取得"""
    normal_dates = set()
    anomaly_dates = set(anomaly_dates)  # 効率的な検索のために集合に変換
    for event_file in event_files:
        events = pd.read_csv(event_file)
        # ファイル形式を判定
        if events.shape[1]==1: type = "line"
        else: type = "range"
        if type == "line":
            # 単一日付イベントの場合
            event_dates = events["event_date"].values
            for date in event_dates:
                event_date = datetime.strptime(date, "%Y-%m-%d")
                start = event_date - timedelta(days=before + range_days)
                end = event_date - timedelta(days=before)
                # 日付範囲を確認し、anomaly_dates に含まれない場合のみ追加
                for day in get_date_range(start, end):
                    if day not in anomaly_dates:
                        normal_dates.add(day)
        elif type == "range":
            # 範囲指定のイベントの場合
            start_dates = events['start_date'].values
            #end_dates = events['end_date']
            for i in range(len(start_dates)):
                start = datetime.strptime(start_dates[i], "%Y-%m-%d") - timedelta(days=before + range_days)
                end = datetime.strptime(start_dates[i], "%Y-%m-%d") - timedelta(days=before)
                # 日付範囲を確認し、anomaly_dates に含まれない場合のみ追加
                for day in get_date_range(start, end):
                    if day not in anomaly_dates:
                        normal_dates.add(day)
    return sorted(list(normal_dates))


def calc_scores(df, event_files, threshold, before_max):
    df['Date'] = df['DATETIME'].dt.date
    #all_dates = get_event_range(event_files,before_max)
    #df = df[df["Date"].isin(anomaly_dates)]
    anomaly_dates_full = get_event_range(event_files,before_max)
    """ df = df[df['Date'].isin(anomaly_dates_full)]
    df["Predicted"] = (df["AnomalyScore"] > threshold).astype(int)
    df_normal = df[~df['Date'].isin(anomaly_dates_full)]
    df_normal["Predicted"] = (df_normal["AnomalyScore"] > threshold).astype(int)
    df_normal["Label"] = 0 """
    normal_dates = get_normal_dates(event_files,before_max,before_max*3,anomaly_dates_full)
    #import pdb;pdb.set_trace()
    df_normal = df[df["Date"].isin(normal_dates)]
    df_normal["Predicted"] = (df_normal["AnomalyScore"] > threshold).astype(int)
    df_normal["Label"] = 0
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for before in range(before_max+1):
        anomaly_dates = get_anomaly_dates(event_files,before)
        #event_dates = get_event_range(event_files,before)
        df_anomaly = df[df["Date"].isin(anomaly_dates)]
        df_anomaly["Predicted"] = (df_anomaly["AnomalyScore"] > threshold).astype(int)
        df_anomaly["Label"] = df_anomaly["Date"].isin(anomaly_dates).astype(int)
        df_all = pd.concat([df_normal,df_anomaly])
        #import pdb; pdb.set_trace()
        precision = '{:.2f}'.format(precision_score(df_all["Label"], df_all["Predicted"]))
        recall = '{:.2f}'.format(recall_score(df_all["Label"], df_all["Predicted"]))
        f1 = '{:.2f}'.format(f1_score(df_all["Label"], df_all["Predicted"]))
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    return precision_scores, recall_scores, f1_scores

def plot_by_date(log_plot, anomaly_score, timestamp, train_range, threshold, img_path, frequency, event_files=None, score_file=None):
    df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": anomaly_score})
    if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
        df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
    # 散布図をプロット
    plt.figure(figsize=(40, 8))
    
    if log_plot:
        plt.yscale('log')
    """df['Date'] = df['DATETIME'].dt.date 
    for date, group in df.groupby('Date'):
        plt.scatter([date]*len(group), group['AnomalyScore'], alpha=0.8, marker='o', color='blue') """
        
    # 外れ値の除去
    num_remove = 15
    # AnomalyScore列の上位num_remove個のインデックスと値を取得
    remove_indices = df.nlargest(num_remove, 'AnomalyScore').index
    """ removed_values = df.loc[remove_indices, 'AnomalyScore']
    # 削除対象のデータを記録
    removed_data = df.loc[remove_indices] """
    # データフレームから削除
    df = df.drop(remove_indices)
    
    plt.scatter(df['DATETIME'], df['AnomalyScore'],alpha=0.8, marker='o', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Scores by Date")
    #plt.xticks(rotation=45)
    plt.grid(True)
    plt.axhline(y=threshold, color='r', linestyle='solid')
    
    # 学習期間/イベント情報の可視化
    color_train_range(train_range)
    if(event_files!=None):
        for file in event_files:
            visualize_events(file)
    
    # 凡例の表示を防止するための設定（重複する日付を削除）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(img_path)
    plt.clf()
    plt.close()
    if(score_file!=None):
        with open(score_file,mode='w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frequency])
            precision = ["precision"]
            recall = ["recall"]
            f1 = ["f1"]
            precision.extend(calc_scores(df,event_files,threshold,before_max=7)[0])
            recall.extend(calc_scores(df,event_files,threshold,before_max=7)[1])
            f1.extend(calc_scores(df,event_files,threshold,before_max=7)[2])
            writer.writerow(precision)
            writer.writerow(recall)
            writer.writerow(f1)