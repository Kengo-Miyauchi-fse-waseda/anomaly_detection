import zipfile
# ZIPファイルを読み込み
zip_f = zipfile.ZipFile('test_pretrain.zip')
# ZIPの中身を取得 
lst = zip_f.namelist() 
# リストから取り出す
for fil in lst:
    print(fil)