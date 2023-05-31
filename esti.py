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
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))     # 回転行列
        self.tvecs = tvecs              # 並進ベクトル

        self.TATE = TATE                # 検出する交点の縦の数
        self.YOKO = YOKO                # 検出する交点の横の数
        self.imgpoints = imgpoints      # 交点の画像座標

        self.camera_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # カメラ原点のワールド座標        Ｗ = Ｒ^T (Ｃ - ｔ)

        self.click_n_w = []

    def onMouse(self, event, u, v, flags, params):      # クリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:              # 画像を左クリックしたら
            self.click_n_w = self.i_to_n_w(u,v)

        if event == cv2.EVENT_MBUTTONDOWN:              # 画像をホイールクリックしたら
            self.i_to_fixed_w(u,v,'z',0)

    def undist_point(self, dist_u, dist_v):
        dist_uv = np.array([dist_u, dist_v],dtype='float32')
        undist_uv = cv2.undistortPoints(dist_uv, self.mtx, self.dist, P=self.mtx)
        undist_uv = undist_uv[0][0]
        return undist_uv
    
    def i_to_n_w(self, u, v):    # ある点の画像座標から，その点の正規化画像座標系における点のワールド座標を求める
        pts_i = self.undist_point(u,v)
        pts_n_x = (pts_i[0] - self.mtx[0][2]) / self.mtx[0][0]
        pts_n_y = (pts_i[1] - self.mtx[1][2]) / self.mtx[1][1]
        pts_n = [[pts_n_x], [pts_n_y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
        pts_n_w = (np.linalg.inv(self.R)) @ (np.array(pts_n) - np.array(self.tvecs))
        return pts_n_w
    
    def i_to_fixed_w(self,u, v, fixed_var, value):      # ワールド座標の三変数のうち，1つを固定することで，1台のカメラからワールド座標を推定する関数
        if fixed_var == 'z': value = -value
        pts_i_undist = self.undist_point(u,v)
        
        pts_n_x = (pts_i_undist[0] - self.mtx[0][2]) / self.mtx[0][0]                   # 画像座標系から正規化座標系に投影　原点を真ん中にしてから，焦点距離で割る
        pts_n_y = (pts_i_undist[1] - self.mtx[1][2]) / self.mtx[1][1]
        
        pts_n_c = [[pts_n_x], [pts_n_y], [1]]                                    # 対象物の正規化画像座標系上の点をカメラ座標系で表す
        pts_n_w = (np.linalg.inv(self.R)) @ (np.array(pts_n_c) - np.array(self.tvecs))    # pts_n_cを世界座標系に変換              Ｗ = Ｒ^T (Ｃ - ｔ)
        
        if fixed_var == 'x':
            slopeyx_w = (self.camera_w[0][0] - pts_n_w[0][0])/(self.camera_w[1][0] - pts_n_w[1][0])
            pts_w_y = ((value - self.camera_w[0][0])/slopeyx_w) + self.camera_w[1][0]
            slopezx_w = (self.camera_w[0][0] - pts_n_w[0][0])/(self.camera_w[2][0] - pts_n_w[2][0])
            pts_w_z = ((value - self.camera_w[0][0])/slopezx_w) + self.camera_w[2][0]
            pts_w_y = round(pts_w_y, 4)
            pts_w_z = round(pts_w_z, 4)
            print([value,pts_w_y,-pts_w_z])
        elif fixed_var == 'y':
            slopexy_w = (self.camera_w[1][0] - pts_n_w[1][0])/(self.camera_w[0][0] - pts_n_w[0][0])
            pts_w_x = ((value - self.camera_w[1][0])/slopexy_w) + self.camera_w[0][0]
            slopezy_w = (self.camera_w[1][0] - pts_n_w[1][0])/(self.camera_w[2][0] - pts_n_w[2][0])
            pts_w_z = ((value - self.camera_w[1][0])/slopezy_w) + self.camera_w[2][0]
            pts_w_x = round(pts_w_x, 4)
            pts_w_z = round(pts_w_z, 4)
            print([pts_w_x,value,-pts_w_z])
        elif fixed_var == 'z':
            slopexz_w = (self.camera_w[2][0] - pts_n_w[2][0])/(self.camera_w[0][0] - pts_n_w[0][0])
            pts_w_x = ((value - self.camera_w[2][0])/slopexz_w) + self.camera_w[0][0]
            slopeyz_w = (self.camera_w[2][0] - pts_n_w[2][0])/(self.camera_w[1][0] - pts_n_w[1][0])
            pts_w_y = ((value - self.camera_w[2][0])/slopeyz_w) + self.camera_w[1][0]
            pts_w_x = round(pts_w_x, 4)
            pts_w_y = round(pts_w_y, 4)
            print([pts_w_x,pts_w_y,-value])
            
    def projection(self, w_x, w_y, w_z):
        C = self.R @ [[w_x], [w_y], [w_z]] + self.tvecs
        N = C/C[2]
        I = self.mtx @ N
        i_u = I[0][0]
        i_v = I[1][0]
        return i_u, i_v

class TwoCameras:
    def __init__(self, cam1, cam2):
        self.cam1 = cam1
        self.cam2 = cam2

    def two_lines_to_point(self):
        line1 = np.hstack((self.cam1.camera_w.T, self.cam1.click_n_w.T)).reshape(2, 3)   # 1カメのワールド座標 と 1カメ画像でクリックされた点のワールド座標を通る直線
        line2 = np.hstack((self.cam2.camera_w.T, self.cam2.click_n_w.T)).reshape(2, 3)   # 2カメのワールド座標 と 2カメ画像でクリックされた点のワールド座標を通る直線
        res = distance_2lines(line1, line2)                                  # 2本の直線の最接近点のワールド座標を求める

        res[0] = round(res[0], 4)                                             # 結果の値を四捨五入
        res[1] = round(res[1], 4)
        res[2] = round(res[2], 4)
        print(f'{res}\n')                                                        # 最終結果であるワールド座標を出力





def update_img(self, img):        # エピポーラ線やクリックした点を描画する関数                                                                                                   # 2カメ画像に対しての処理の場合                                                                                              # 1カメ画像に対しての処理の場合
    if self.click_count >= 1:                                                                                   # 画像をクリックしていたら，
        img = cv2.circle(img, (int(self.obj1_i1x),int(self.obj1_i1y)), 4, (0, 165, 255), thickness=-1)          # クリックした点を描画
        if self.click_count == 2:                                                                               # 2カメ画像をクリックしていたら，
            startpoint_i1y, endpoint_i1y = self.epipo(self.camera2_w, self.obj2_w, 2)                           # エピポーラ線を求める
            img = cv2.line(img, (0, int(startpoint_i1y)), (img.shape[1], int(endpoint_i1y)), (0,255,255), 2)    # エピポーラ線を描画
    #img = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
    return img
    

def axes_pts_i(rvecs, tvecs, mtx, dist):         # 座標軸の先端の画像座標を求める関数
    # 軸の定義
    axes_pts_w = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)
    axes_pts_i, _ = cv2.projectPoints(axes_pts_w, rvecs, tvecs, mtx, dist)
    return axes_pts_i

def draw_axes(img, corners, imgpts):         # 座標軸を描画する関数
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 3)   # X軸 Blue
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 3)   # Y軸 Green
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 3)   # Z軸 Red
    return img

def distance_2lines(line1, line2):    # 直線同士の最接近距離と最接近点を求める関数
    '''
    直線同士の最接近距離と最接近点
    return 直線間の距離, line1上の最近接点、line2上の最近接点
    '''
    line1 = [np.array(line1[0]),np.array(line1[1])]
    line2 = [np.array(line2[0]),np.array(line2[1])]

    if abs(np.linalg.norm(line1[1]))<0.0000001:
        return None,None,None
    if abs(np.linalg.norm(line2[1]))<0.0000001:
        return None,None,None

    p1 = line1[0]
    p2 = line2[0]

    v1 = line1[1] / np.linalg.norm(line1[1])
    v2 = line2[1] / np.linalg.norm(line2[1])

    d1 = np.dot(p2 - p1,v1)
    d2 = np.dot(p2 - p1,v2)
    dv = np.dot(v1,v2)

    if (abs(abs(dv) - 1) < 0.0000001):
        v = np.cross(p2 - p1,v1)
        return np.linalg.norm(v),None,None

    t1 = (d1 - d2 * dv) / (1 - dv * dv)
    t2 = (d2 - d1 * dv) / (dv * dv - 1)

    #外挿を含む最近接点
    q1 = p1 + t1 * v1
    q2 = p2 + t2 * v2

    q1[0]=-q1[0]
    q1[1]=-q1[1]

    q2[0]=-q2[0]
    q2[1]=-q2[1]

    # XYZ座標の候補が2つあるため，平均をとる
    q3x = (q1[0]+q2[0])
    q3y = (q1[1]+q2[1])
    q3z = (q1[2]+q2[2])
    
    #return np.linalg.norm(q2 - q1), q1, q2
    return ([q3x, q3y, q3z])


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
    

    # カメラの設定
    cam_id = [0,2]
    cap_list = []
    for i in cam_id:
        cap_list.append(cv2.VideoCapture(i))

    corners_cam1 = np.array([102, 171, 173, 157, 247, 143, 318, 132, 389, 121, 452, 113, 103, 220, 182, 205, 261, 191, 339, 177, 415, 164, 482, 153, 107, 281, 193, 264, 280, 247, 366, 230, 446, 215, 517, 200, 114, 353, 207, 335, 304, 315, 396, 295, 482, 275, 556, 257, 125, 437, 228, 419, 332, 395, 432, 370, 522, 343, 598, 320],dtype='float32').reshape(-1,1,2)
    corners_cam2 = np.array([253, 79, 316, 93, 381, 109, 453, 127, 525, 145, 595, 166, 218, 117, 283, 133, 355, 150, 431, 172, 510, 193, 588, 215, 179, 161, 247, 179, 322, 201, 405, 225, 491, 249, 577, 275, 135, 213, 204, 236, 285, 261, 372, 290, 467, 319, 560, 346, 85, 271, 157, 300, 240, 332, 334, 366, 435, 400, 538, 430],dtype='float32').reshape(-1,1,2)
    corners_list = [corners_cam1, corners_cam2]


    imgpoints_list = []
    # カメラからの画像取得
    for i, (cap, conners) in enumerate(zip(cap_list, corners_list)):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        SubPix_corners = cv2.cornerSubPix(gray, conners, (11,11), (-1,-1), criteria)
        imgpoints_list.append(SubPix_corners)

        cv2.imshow(f'camera{i+1}', frame)       # あとでクリックイベントを設定するために，ここでウィンドウを出しておく


    # 内部パラメータ
    mtx1 = np.array([670.42674146, 0, 343.11230192, 0, 671.2945385, 228.79870936, 0, 0, 1]).reshape(3,3)
    mtx2 = np.array([679.96806792, 0, 346.17428752, 0, 680.073095, 223.11864352, 0, 0, 1]).reshape(3,3)
    mtx_list = [mtx1, mtx2]

    # 歪み係数
    dist1 = np.array([-0.44761361, 0.30931205, -0.00102743, 0.00163647, -0.1970753]).reshape(1,-1)
    dist2 = np.array([-4.65322789e-01, 3.88192556e-01, -2.58061417e-03, -1.69216076e-04, -3.97886097e-01]).reshape(1,-1)

    dist_list = [dist1, dist2]

    # 外部パラメータ
    rvecs_list = []
    tvecs_list = []
    axes_i_list = []
    for SubPix_corners, mtx, dist, corners in zip(imgpoints_list, mtx_list, dist_list,   corners_list):
        # 外部パラメータ
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, SubPix_corners, mtx, dist)
        rvecs_list.append(rvecs)
        tvecs_list.append(tvecs)
        # 座標軸の画像座標
        axes_i = axes_pts_i(rvecs, tvecs, mtx, dist)
        #axes_i = np.append(axes_i, corners[0][0][0])
        axes_i_list.append(axes_i)

    """
    ret：
    mtx：camera matrix，内部パラメータ
    dist：distortion coefficients，歪み係数
    rvecs：rotation vectors，外部パラメータの回転ベクトル
    tvecs：translation vectors，外部パラメータの並進ベクトル
    """

    cam_list = []
    for i, (mtx, dist, rvecs, tvecs, imgpoints) in enumerate(zip(mtx_list, dist_list, rvecs_list, tvecs_list, imgpoints_list)):
        cam = Camera(mtx, dist, rvecs, tvecs, TATE, YOKO, imgpoints)
        cam_list.append(cam)
        cv2.setMouseCallback(f'camera{i+1}', cam.onMouse)         # 1カメの画像に対するクリックイベント

    while True:
        for i, (cam, cap, conners, axes_i) in enumerate(zip(cam_list, cap_list, corners_list, axes_i_list)):
            _, frame = cap.read()
            img_axes = draw_axes(frame, conners, axes_i)
            cv2.imshow(f'camera{i+1}', img_axes)


        #繰り返し分から抜けるためのif文
        key =cv2.waitKey(1)
        if key == 27:   #Escで終了
            cv2.destroyAllWindows()
            break
        elif key == ord('e'):
            tcams = TwoCameras(cam_list[0],cam_list[1])
            tcams.two_lines_to_point()

    for cap in cap_list:
        cap.release()

if __name__ == "__main__":
    main()

