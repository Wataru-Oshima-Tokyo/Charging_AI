from PIL import Image
import numpy as np
import os,glob

# クラスラベル
labels = []
basename="outlet"
for i in range(474):
    exp = basename +"_"
    if i <10:
        exp +="00" + str(i)
    elif i<100:
        exp +="0" + str(i)
    else:
        pass
    labels.append(exp)

# ディレクトリ
dataset_dir = "../models/dataset.npy" # 前処理済みデータ
model_dir   = "../models/cnn_h5"      # 学習済みモデル
# リサイズ設定
resize_settings = (50,50)

# 画像データ
X_train = [] # 学習
y_train = [] # 学習ラベル
X_test  = [] # テスト
y_test  = [] # テストラベル

for class_num, label in enumerate(labels):
    
    # 写真のディレクトリ
    photos_dir = "../models/outlet/" + label
    
    # 画像データを取得
    files = glob.glob(photos_dir + "/*.jpg")
    
    #写真を順番に取得
    for i,file in enumerate(files):
        
        # 画像を1つ読込
        image = Image.open(file)
        
        # 画像をRGBの3色に変換
        image = image.convert("RGB")
        
        # 画像のサイズを揃える
        image = image.resize(resize_settings)
        
        # 画像を数字の配列変換
        data  = np.asarray(image) 

        # テストデータ追加
        if i%4 ==0:
            X_test.append(data)
            y_test.append(class_num)
            
        # 学習データ傘増し
        else:            
            # -20度から20度まで4度刻みで回転したデータを追加
            for angle in range(-25,20,5):
                # 回転
                img_r = image.rotate(angle)
                # 画像 → 数字の配列変換
                data  = np.asarray(img_r)
                # 追加
                X_train.append(data)
                y_train.append(class_num)
                # 画像左右反転
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data      = np.asarray(img_trans)
                X_train.append(data)
                y_train.append(class_num)        
        
        
# X,YがリストなのでTensorflowが扱いやすいようnumpyの配列に変換
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)


# 前処理済みデータを保存
dataset = (X_train,X_test,y_train,y_test)
np.save(dataset_dir,dataset)
