


def main(path):
    X     = []                               # 推論データ格納
    image = Image.open(path)                 # 画像読み込み
    image = image.convert("RGB")             # RGB変換
    image = image.resize(resize_settings)    # リサイズ
    data  = np.asarray(image)                # 数値の配列変換
    X.append(data)
    X     = np.array(X)
    
    # モデル呼び出し
    model = predict()
    
    # numpy形式のデータXを与えて予測値を得る
    model_output = model.predict([X])[0]
    # 推定値 argmax()を指定しmodel_outputの配列にある推定値が一番高いインデックスを渡す
    predicted = model_output.argmax()
    # アウトプット正答率
    accuracy = int(model_output[predicted] *100)
    print("{0} ({1} %)".format(labels[predicted],accuracy))
