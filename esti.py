import numpy as np
import cv2
import math
import time

class Camera:
    def __init__(self, mtx, dist, rvecs, tvecs, TATE, YOKO, imgpoints):
        self.mtx = mtx                  # 内部パラメータ
        self.dist = dist                # 歪み係数
        self.rvecs = rvecs              # 回転ベクトル
        # 回転ベクトルを3×1から3×3に変換
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))     # 1カメの回転行列
        self.tvecs = tvecs              # 並進ベクトル

        self.TATE = TATE                # 検出する交点の縦の数
        self.YOKO = YOKO                # 検出する交点の横の数
        self.imgpoints = imgpoints      # 交点の画像座標

        self.camera_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # カメラ原点のワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)

    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:                                              # 画像を左クリックしたら，
            click_n_w = self.i_to_n_w(x,y)

        if event == cv2.EVENT_MBUTTONDOWN:
            pass

    def undist_point(self, dist_u, dist_v):
        dist_uv = np.array([dist_u, dist_v],dtype='float32')
        undist_uv = cv2.undistortPoints(dist_uv, self.mtx, self.dist, P=self.mtx)
        undist_uv = dist_uv[0][0]
        undist_uv = [dist_uv[0],dist_uv[1]]
        return undist_uv

    def i_to_n_w(self, u, v):    # ある点の画像座標から，その点の正規化画像座標系における点のワールド座標を求める
        pts_i = self.undist_point(u,v)
        pts_n_x = (pts_i[0] - self.mtx[0][2]) / self.mtx[0][0]
        pts_n_y = (pts_i[1] - self.mtx[1][2]) / self.mtx[1][1]
        pts_n = [[pts_n_x], [pts_n_y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
        pts_n_w = (np.linalg.inv(self.R)) @ (np.array(pts_n) - np.array(self.tvecs))
        return pts_n_w
    



def main():
    # 検出する交点の数
    TATE = 5
    YOKO = 6

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((TATE*YOKO,3), np.float32)
    objp[:,:2] = np.mgrid[0:YOKO,0:TATE].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space

    objpoints.append(objp)      # object point
    

    # 軸の定義
    axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)

    # カメラの設定
    cam_id = [0,2]
    cap_list = []
    for i in cam_id:
        cap_list.append(cv2.VideoCapture(i))

    corners_cam1 = np.array([253, 79, 316, 93, 381, 109, 453, 127, 525, 145, 595, 166, 218, 117, 283, 133, 355, 150, 431, 172, 510, 193, 588, 215, 179, 161, 247, 179, 322, 201, 405, 225, 491, 249, 577, 275, 135, 213, 204, 236, 285, 261, 372, 290, 467, 319, 560, 346, 85, 271, 157, 300, 240, 332, 334, 366, 435, 400, 538, 430],dtype='float32').reshape(-1,1,2)
    corners_cam2 = np.array([102, 171, 173, 157, 247, 143, 318, 132, 389, 121, 452, 113, 103, 220, 182, 205, 261, 191, 339, 177, 415, 164, 482, 153, 107, 281, 193, 264, 280, 247, 366, 230, 446, 215, 517, 200, 114, 353, 207, 335, 304, 315, 396, 295, 482, 275, 556, 257, 125, 437, 228, 419, 332, 395, 432, 370, 522, 343, 598, 320],dtype='float32').reshape(-1,1,2)
    corners_list = [corners_cam1, corners_cam2]


    imgpoints_list = []
    # カメラからの画像取得
    for cap, conners in zip(cap_list, corners_list):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        SubPix_corners = cv2.cornerSubPix(gray, conners, (11,11), (-1,-1), criteria)
        imgpoints_list.append(SubPix_corners)

    # 内部パラメータ
    mtx = np.array([679.96806792, 0, 346.17428752, 0, 680.073095, 223.11864352, 0, 0, 1]).reshape(3,3)
    mtx2 = np.array([670.42674146, 0, 343.11230192, 0, 671.2945385, 228.79870936, 0, 0, 1]).reshape(3,3)
    mtx_list = [mtx, mtx2]

    # 歪み係数
    dist = np.array([-4.65322789e-01, 3.88192556e-01, -2.58061417e-03, -1.69216076e-04, -3.97886097e-01]).reshape(1,-1)
    dist2 = np.array([-0.44761361, 0.30931205, -0.00102743, 0.00163647, -0.1970753]).reshape(1,-1)
    dist_list = [dist, dist2]

    # 外部パラメータ
    rvecs_list = []
    tvecs_list = []
    axis_i_list = []
    for SubPix_corners, mtx, dist in zip(imgpoints_list, mtx_list, dist_list):
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, SubPix_corners, mtx, dist)
        rvecs_list.append(rvecs)
        tvecs_list.append(tvecs)
        axis_imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        axis_i_list.append(axis_imgpts)

    """
    ret：
    mtx：camera matrix，カメラ行列(内部パラメータ)
    dist：distortion coefficients，レンズ歪みパラメータ
    rvecs：rotation vectors，回転ベクトル
    tvecs：translation vectors，並進ベクトル
    """

if __name__ == "__main__":
    main()

