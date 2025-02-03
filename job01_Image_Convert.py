# 250203
# CNN 실습 - 말과 사람 이중 분류기
# 작업01 - 이미지 처리

import glob                           # 파일을 다룰 때 사용하는 파이썬 기본 패키지
import numpy as np
from PIL import Image                 # pillow
from sklearn.model_selection import train_test_split

img_dir = './Datasets/'               # 이미지 경로
categories = ['horse*', 'human*']     # 분류할 태그

image_w = 150                         # 이미지 사이즈 지정 / 커질수록 성능이 좋아지지만 연산이 오래걸림
image_h = 150                         # 300 x 300 => 150 x 150

pixel = image_w * image_h             # 픽셀 총량 계산
X = []                                # 빈 리스트 생성
Y = []
files = None


for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '*.png')     # horse으로 한번 human로 한번 라벨링 (0, 1)
    for i, f in enumerate(files):
        try:
            # 이미지 변환
            img = Image.open(f)                         # 파일 오픈
            img = img.convert('RGB')                    # R, G, B 3색으로 변경
            img = img.resize((image_w, image_h))        # 150 x 150 크기로 리사이징
            data = np.asarray(img)                      # array와 동일하지만 원본이 수정되면 asarray 복사본도 같이 수정된다.

            # 변환한 이미지 정리
            X.append(data)
            Y.append(idx)                               # horse Y = 0 / human Y == 라벨링

            if i % 300 == 0:                            # 이미지 300개 진행될 때 마다 print
                print(category, ':', f)

        except:                                         # 예외처리 (에러나면 수행)
            print(category, i, '번에서 에러')

X = np.array(X)                                         # X에는 이미지 데이터
Y = np.array(Y)                                         # Y에는 카테고리
X = X / 255

print(X[0])
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.1)                        # 테스트 사이즈 = 10%는 테스트용으로 빼둔다.

# xy = (X_train, X_test, Y_train, Y_test)
# np.save('../datasets/binary_image_data.py', xy)       # 구버전 numpy로는 사용가능

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 가공된 이미지 저장
np.save('./Trains/binary_X_train.npy', X_train)
np.save('./Trains/binary_X_test.npy', X_test)
np.save('./Trains/binary_Y_train.npy', Y_train)
np.save('./Trains/binary_Y_test.npy', Y_test)


