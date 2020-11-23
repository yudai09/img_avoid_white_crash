import cv2
import numpy as np
import glob

ref_img = cv2.imread('img/ref.jpg', cv2.IMREAD_GRAYSCALE)
warped_img_list = []

# レジストレーションは[https://qiita.com/suuungwoo/items/9598cbac5adf5d5f858e:embed:cite]を参照した。
for float_img_path in glob.glob('img/image*.jpg'):
    akaze = cv2.AKAZE_create()
    print(float_img_path)
    float_img = cv2.imread(float_img_path, cv2.IMREAD_GRAYSCALE)
    float_img_color = cv2.imread(float_img_path, cv2.IMREAD_COLOR)
    float_kp, float_des = akaze.detectAndCompute(float_img, None)
    ref_kp, ref_des = akaze.detectAndCompute(ref_img, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(float_des, ref_des, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # 適切なキーポイントを選択
    ref_matched_kpts = np.float32(
        [float_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32(
        [ref_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィを計算
    H, status = cv2.findHomography(
        ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

    if H is None:
        print('skip')
        continue

    # 画像を変換
    warped_image = cv2.warpPerspective(
        float_img_color, H, (float_img.shape[1], float_img.shape[0]))

    warped_img_list.append(warped_image)

warped_img_stacked = np.stack([warped_img_list])
warped_img_mean = np.mean(warped_img_stacked, axis=1)
warped_img_mean = warped_img_mean.astype(np.uint8)
warped_img_mean = warped_img_mean.squeeze(0)

cv2.imwrite("warped_img_mean.jpg", warped_img_mean)