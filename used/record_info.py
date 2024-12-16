from datetime import datetime

def recordInfo(out_dir, info):
    file_path = out_dir + "/info.txt"  # ディレクトリ + ファイル名

    # 現在の時刻を取得
    current_time = datetime.now()

    # 現在の時刻をフォーマット（例: "YYYY-MM-DD HH:MM:SS"）
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # 指定したパスでファイルを作成し、時刻を書き込む
    with open(file_path, "w") as file:
        file.write(f"End Time : {formatted_time}\n")
        for x in info:
            file.write(x[0]+" : "+str(x[1])+"\n")
