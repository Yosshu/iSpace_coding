#!/usr/bin/env python

import numpy as np
import cv2
aruco = cv2.aruco
import glob
import pyautogui
import copy
import math
import colorsys
from numpy import linalg as LA
from itertools import product

import torch
import time


def draw(img, corners, imgpts):         # 座標軸を描画する関数
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 3)   # X軸 Blue
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 3)   # Y軸 Green
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 3)   # Z軸 Red
    return img

def axes_check(img, tate, yoko, objp, criteria, axis):      # Z軸が正しく伸びているかを確認するための関数
    objpoints0 = []
    imgpoints0 = []
    gray0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret0, corners0 = cv2.findChessboardCorners(gray0, (yoko,tate),None)
    if ret0 == True:
        objpoints0.append(objp)      # object point
        corners02 = cv2.cornerSubPix(gray0,corners0,(11,11),(-1,-1),criteria) # 精度を上げている
        imgpoints0.append(corners02)
        cv2.waitKey(500)
        ret, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints0, imgpoints0, gray0.shape[::-1],None,None)
        # Find the rotation and translation vectors.
        _, rvecs0, tvecs0, _ = cv2.solvePnPRansac(objp, corners02, mtx0, dist0)

        imgpts0, _ = cv2.projectPoints(axis, rvecs0, tvecs0, mtx0, dist0)
        return ret0, corners02, imgpts0
    return ret0, None, None

def drawpoints(img, points,b,g,r):
    for i in points:
        if 0<=i[0]<img.shape[1] and 0<=i[1]<img.shape[0]:
            img = cv2.circle(img, (int(i[0]),int(i[1])), 2, (b, g, r), thickness=-1)
    return img


def nearest(ret,a,b):
    if ret:
        na, nb = len(a), len(b)
        ## Combinations of a and b
        comb = product(range(na), range(nb))
        ## [[distance, index number(a), index number(b)], ... ]
        l = [[np.linalg.norm(a[ia] - b[ib]), ia, ib] for ia, ib in comb]
        ## Sort with distance
        l.sort(key=lambda x: x[0])
        _, ia, ib = l[0]
        return a[ia], b[ib]
    return None,None

def tangent_angle(u: np.ndarray, v: np.ndarray):        # 2つのベクトルのなす角を求める関数
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n

    output = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

    w = np.cross(u, v)
    if w < 0:
        output = -output

    return output


def isint(s):  # 整数値を表しているかどうかを判定
    try:
        int(s, 10)  # 文字列を実際にint関数で変換してみる
    except ValueError:
        return False  # 例外が発生＝変換できないのでFalseを返す
    else:
        return True  # 変換できたのでTrueを返す

def isfloat(s):  # 浮動小数点数値を表しているかどうかを判定
    try:
        float(s)  # 文字列を実際にfloat関数で変換してみる
    except ValueError:
        return False  # 例外が発生＝変換できないのでFalseを返す
    else:
        return True  # 変換できたのでTrueを返す


"""def draw_map(xyz, width, height, xmin, xmax, ymin, ymax):        # 環境地図を表示する関数
    # 環境地図の初期化（サイズや背景色などを指定）
    map_image = np.zeros((height, width, 3), dtype=np.uint8)
    map_image.fill(255)  # 白色で塗りつぶす

    # 人のワールド座標のXとY座標を地図上に描画
    for point in xyz:
        x, y, _ = point
        # 座標変換して描画
        x_on_map = int((x - xmin) / (xmax - xmin) * width)
        y_on_map = int((y - ymin) / (ymax - ymin) * height)
        cv2.circle(map_image, (x_on_map, y_on_map), radius=5, color=(0, 0, 255), thickness=-1)

    # 地図の表示
    cv2.imshow("Environment Map", map_image)"""


def draw_map(xyz, width, height, xmin, xmax, ymin, ymax):
    # 環境地図の初期化（サイズや背景色などを指定）
    map_image = np.zeros((height, width, 3), dtype=np.uint8)
    map_image.fill(255)  # 白色で塗りつぶす

    # 人のワールド座標のXとY座標を地図上に描画
    for point in xyz:
        x, y, _ = point
        x = x*50
        y = y*50
        # 座標変換して描画
        x_on_map = int(((x - xmin) / (xmax - xmin)) * width)
        y_on_map = int(((y - ymin) / (ymax - ymin)) * height)
        cv2.circle(map_image, (x_on_map, y_on_map), radius=5, color=(0, 0, 255), thickness=-1)

    # 格子状の線を描画
    grid_spacing_cm = 50  # 格子の間隔（50cmごと）
    grid_spacing_pixels = int((grid_spacing_cm / (xmax - xmin)) * width)  # 格子の間隔をピクセルに変換
    grid_color = (128, 128, 128)  # 格子の色（グレー）
    grid_thickness = 1  # 格子の太さ

    # 縦の格子線を描画
    for x in range(0, width, grid_spacing_pixels):
        cv2.line(map_image, (x, 0), (x, height), grid_color, grid_thickness)

    # 横の格子線を描画
    for y in range(0, height, grid_spacing_pixels):
        cv2.line(map_image, (0, y), (width, y), grid_color, grid_thickness)

    cv2.rectangle(map_image, (200,100), (700,300), (50,50,0), 2)

    # 地図の表示
    cv2.imshow("Environment Map", map_image)

(0,1),(5,3)

class Estimation:
    def __init__(self, mtx, dist, rvecs, tvecs, img, imgpoints, tate, yoko):
        self.mtx = mtx                  # 1カメの内部パラメータ
        self.dist = dist                # 1カメの歪み係数
        self.rvecs = rvecs              # 1カメの回転ベクトル
        self.tvecs = tvecs              # 1カメの並進ベクトル


        self.imgpoints = imgpoints      # 1カメで見つかったチェッカーボードの交点の画像座標
        self.tate = tate                # 検出するチェッカーボードの交点の縦の数
        self.yoko = yoko                # 検出するチェッカーボードの交点の横の数
        
        # 回転ベクトルを3×1から3×3に変換
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))     # 1カメの回転行列

        self.LRMclick = None    # 左右中クリックの判断

        # クラス内の関数間で共有したい変数
        self.obj1_i1x = 0               # 1カメでクリックした点の1カメ画像座標
        self.obj1_i1y = 0
        self.img = img      # 軸だけ描画された1カメの画像
        self.pn = [1, 1, 1]             # 1 or -1，ワールド座標がプラスかマイナスか，出力する直前にかける [X軸座標, Y軸座標, Z軸座標]
        #self.origin = []

        self.camera1_w = []             # 1カメのワールド座標
        self.obj1_w = []                # 1カメでクリックした点のワールド座標


        self.target_i = []  # 左or右クリックした点の画像座標
        self.target_w = []  # 左or右クリックした点のワールド座標

        self.scale = 50      # 1マス50cm

        self.robot_w = []
        self.robot_vector = []
        self.ret_robot = False
        self.input_angle = 0
        self.robotcam_height = 22.5
        self.robotcam_len = 16
        
        self.click = 0
        self.widthy = 1

    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if self.click == 0:
            if event == cv2.EVENT_MBUTTONDOWN:    # 中クリック
                self.click = 1
                res = self.pointFixZ(x,y,0)
                res = [round(n*self.scale,2) for n in res]
                print(f"{res} [cm]")
                self.obj1_i1x = x
                self.obj1_timekeepingick = 'M'
                self.click = 0


    def line_SEpoint(self, x, y, num):      # 始点（カメラ）と終点（正規化画像座標）のワールド座標を求める関数，numは1カメか2カメか
        obj_i = [x,y]
        if num == 1:
            obj_sphi1 = self.undist_pts(np.array([obj_i],dtype='float32'),1)

            obj_n1x = (obj_sphi1[0] - self.mtx[0][2]) / self.mtx[0][0]                   # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n1y = (obj_sphi1[1] - self.mtx[1][2]) / self.mtx[1][1]
            
            obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
            obj1_w = (np.linalg.inv(self.R)) @ (np.array(obj_n1) - np.array(self.tvecs))
            #obj1_w = (self.R.T) @ (np.array(obj_n1) - np.array(self.tvecs))                        # obj_n1を世界座標系に変換              Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            
            #camera1_w = (np.linalg.inv(R)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))     # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            camera1_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)

            return camera1_w, obj1_w

        elif num == 2:
            obj_sphi2 = self.undist_pts(np.array([obj_i],dtype='float32'),2)

            obj_n2x = (obj_sphi2[0] - self.mtx2[0][2]) / self.mtx2[0][0]                 # 対象物の2カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n2y = (obj_sphi2[1] - self.mtx2[1][2]) / self.mtx2[1][1]
            
            obj_n2 = [[obj_n2x], [obj_n2y], [1]]                                    # 対象物の2カメ正規化画像座標系を2カメカメラ座標系に変換
            #obj1_w = (np.linalg.inv(R)) @ (np.array(obj_n1) - np.array(tvecs))
            obj2_w = (self.R2.T) @ (np.array(obj_n2) - np.array(self.tvecs2))                       # obj_n2を世界座標系に変換              Ｗ = Ｒ2^T (Ｃ2 - ｔ2)
            
            #camera2_w = (np.linalg.inv(R2)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))        # 2カメのワールド座標        Ｗ = Ｒ2^T (Ｃ2 - ｔ2)
            camera2_w = (self.R2.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))           # 2カメのワールド座標        Ｗ = Ｒ2^T (Ｃ2 - ｔ2)

            return camera2_w, obj2_w

        return None, None


    def line_update(self, img):        # エピポーラ線やクリックした点を描画する関数
        if self.LRMclick == 'L':
            img = cv2.circle(img, (int(self.target_i[0]),int(self.target_i[1])), 2, (0, 165, 255), thickness=-1)          # 左クリックした点を描画
        elif self.LRMclick == 'R2':
            img = cv2.circle(img, (int(self.target_i[0]),int(self.target_i[1])), 2, (255, 165, 0), thickness=-1)          # 右クリックした点を描画
        elif self.LRMclick == 'M':
            img = cv2.circle(img, (int(self.obj1_i1x),int(self.obj1_i1y)), 2, (255,0,255), thickness=-1)          # 中クリックした点を描画
        return img


    def undist_pts(self, pts_uv, num):
        if num == 1:
            pts_uv = cv2.undistortPoints(pts_uv, self.mtx, self.dist, P=self.mtx)
            pts_uv = pts_uv[0][0]
            pts_uv = [pts_uv[0],pts_uv[1]]
        return pts_uv

    def pointFixZ(self,ix,iy,wz):
        #floor_wz = 0
        obj_i = [ix,iy]
        floor_wz = -wz

        obj_sphi1 = self.undist_pts(np.array([obj_i],dtype='float32'),1)

        obj_n1x = (obj_sphi1[0] - self.mtx[0][2]) / self.mtx[0][0]                   # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
        obj_n1y = (obj_sphi1[1] - self.mtx[1][2]) / self.mtx[1][1]
        
        obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
        obj1_w = (np.linalg.inv(self.R)) @ (np.array(obj_n1) - np.array(self.tvecs))
        #obj1_w = (self.R.T) @ (np.array(obj_n1) - np.array(self.tvecs))                        # obj_n1を世界座標系に変換              Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
        
        camera1_w = (np.linalg.inv(self.R)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))     # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
        #camera1_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
        
        slopexz_w = (camera1_w[0][2][0] - obj1_w[0][2][0])/(camera1_w[0][0][0] - obj1_w[0][0][0])
        floor_wx = ((floor_wz - camera1_w[0][2][0])/slopexz_w) + camera1_w[0][0][0]
        slopeyz_w = (camera1_w[0][2][0] - obj1_w[0][2][0])/(camera1_w[0][1][0] - obj1_w[0][1][0])
        floor_wy = ((floor_wz - camera1_w[0][2][0])/slopeyz_w) + camera1_w[0][1][0]
        floor_wx = round(floor_wx, 4)
        floor_wy = round(floor_wy, 4)

        return [floor_wx,floor_wy,-floor_wz]

    """
    def getTarget(self):
        ret = False
        if self.Lclick_count == 1:
            ret = True
        return ret, self.target_i
    """

    def in_area(self, bottom_xy_w , area_xmin_w, area_ymin_w, area_xmax_w, area_ymax_w):
        if area_xmin_w <= bottom_xy_w[0] <= area_xmax_w and area_ymin_w <= bottom_xy_w[1] <= area_ymax_w:
            return True
        return False
    
    def draw_area(self, img, Xw1, Yw1, Xw2, Yw2, Zw, flag, count, total_time):
        points_w = np.float32([[Xw1, Yw1, Zw], [Xw2, Yw1, Zw], [Xw1, Yw2, Zw], [Xw2, Yw2, Zw]]).reshape(-1,3)
        points_i, _ = cv2.projectPoints(points_w, self.rvecs[-1], self.tvecs[-1], self.mtx, self.dist)
        points_i = points_i.reshape(-1, 2)
        points_i = np.asarray(points_i, dtype = int)
        points_i = points_i[[0, 1, 3, 2],:]
        area_color = (255,255,0) if flag else (50,50,0)
        cv2.polylines(img, [points_i], True, area_color, thickness=3)
        cv2.putText(img,                                                                    # エリア内の人の数
            text= f'{count}',
            org=(int(np.mean(points_i[:, 0])-20), int(np.mean(points_i[:, 1])-10)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3.0,
            color=area_color,
            thickness=3,
            lineType=cv2.LINE_4)
        cv2.putText(img,                                                                    # 滞在時間
            text= f'{round(total_time,2)}s',
            org=(int(points_i[0][0]), int(np.mean(points_i[:, 1])+50)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3.0,
            color=area_color,
            thickness=3,
            lineType=cv2.LINE_4)
        return img
    
class RetainingTime:
    def __init__(self):
        self.total_time = 0
        self.previous_time = 0
        self.start_time = 0
        self.time_started = False

    def start_and_accumulate(self, flag):
        if flag == True:
            if self.time_started == False:
                # 時間計測開始
                self.start_time = time.perf_counter()
                self.time_started = True
            elif self.time_started == True:
                dtime = self.dtime()
                self.total_time = dtime + self.previous_time
        elif flag == False and self.time_started == True:
            dtime = self.dtime()
            self.previous_time = dtime + self.previous_time
            self.time_started = False
        return self.total_time
    
    def dtime(self):
        end_time = time.perf_counter()
        # 経過時間（秒）
        dtime = end_time - self.start_time
        return dtime
        


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')     # モデルの読み込み
    model.classes = [0]                                         # モデルを人のみに限定する

    # 検出するチェッカーボードの交点の数
    tate = 5
    yoko = 6
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((tate*yoko,3), np.float32)
    objp[:,:2] = np.mgrid[0:yoko,0:tate].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 軸の定義
    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)


    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()
    # CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

    cap = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0

    # カメラの解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 幅の設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 高さの設定



    _, frame1 = cap.read()           #カメラからの画像取得
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera1' , frame1)
    corners = np.array([389, 266, 494, 265, 605, 262, 719, 261, 830, 261, 937, 262, 365, 338, 478, 337, 600, 336, 724, 335, 846, 334, 962, 333, 338, 423, 461, 424, 593, 424, 730, 423, 864, 421, 990, 417, 309, 523, 441, 529, 586, 531, 736, 531, 885, 526, 1018, 517, 279, 637, 422, 651, 578, 657, 743, 657, 905, 650, 1050, 635],dtype='float32').reshape(-1,1,2)
    corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners12)

    objpoints.append(objp)      # object point

    # パラメータの表示
    # Draw and display the corners
    #img = cv2.drawChessboardCorners(img, (yoko,tate), corners12,ret)
    #img2 = cv2.drawChessboardCorners(img2, (yoko,tate), corners22,ret2)
    #cv2.imshow('drawChessboardCorners',img)
    cv2.waitKey(500)
    """
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None,flags=cv2.CALIB_RATIONAL_MODEL) # _, _ は，rvecs, tvecs
    """
    mtx = np.array([1.01644397e+03, 0.00000000e+00, 6.87903319e+02, 0.00000000e+00, 1.01960682e+03, 3.37505807e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape(3,3)

    dist = np.array([-4.31940522e-01, 2.19409920e-01, -4.78545150e-05, 1.18269452e-03, -7.07910142e-02])

    _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners12, mtx, dist)
    rvecs = [rvecs]
    tvecs = [tvecs]

    """
    ret：
    mtx：camera matrix，カメラ行列(内部パラメータ)
    dist：distortion coefficients，レンズ歪みパラメータ
    rvecs：rotation vectors，回転ベクトル
    tvecs：translation vectors，並進ベクトル
    """
    
    # project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis, rvecs[-1], tvecs[-1], mtx, dist)
    
    es = Estimation(mtx, dist, rvecs, tvecs, frame1, imgpoints, tate, yoko)
    rtime = RetainingTime()

    cv2.setMouseCallback('camera1', es.onMouse)         # 1カメの画像に対するクリックイベント

    WHERE_AREA = ((0,1),(5,3))

    # 環境地図の範囲指定
    MAP_XMIN, MAP_XMAX = -100, 350
    MAP_YMIN, MAP_YMAX = 0, 200
    # 環境地図のサイズ指定
    MAP_WIDTH, MAP_HEIGHT = 900, 400


    while True:
        ret, frame1 = cap.read()           # カメラからの画像取得
        results = model(frame1)             # 人の検出
        img_axes = frame1.copy()
        #objects = results.pandas().xyxy[0]
        """
        xmins = results.pandas().xyxy[0]['xmin']
        ymins = results.pandas().xyxy[0]['ymin']
        xmaxs = results.pandas().xyxy[0]['xmax']
        ymaxs = results.pandas().xyxy[0]['ymax']
        """
        #print(f'xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}')
        count_people = 0
        any_in_area = False
        people_bottom_w_xyz_list = []
        for count, bbox in enumerate(results.xyxy[0]):
            xmin, ymin, xmax, ymax, conf, cls = bbox
            person_bottom_i_x = int((xmin+xmax)/2)
            person_bottom_i_y = int(ymax)
            people_bottom_w_xyz = es.pointFixZ(person_bottom_i_x, person_bottom_i_y, 0)
            people_bottom_w_xyz_list.append(people_bottom_w_xyz)
            is_in_area = es.in_area(people_bottom_w_xyz, *WHERE_AREA[0], *WHERE_AREA[1])
            any_in_area = any_in_area or is_in_area
            if is_in_area:
                count_people = count_people + 1
            
            PERSON_COLOR = (0,0,200)
            # バウンディングボックスの描画
            if conf >= 0.5:
                if int(cls) == 0:  # クラスがpersonの場合
                    cv2.rectangle(img_axes, (int(xmin), int(ymin)), (int(xmax), int(ymax)), PERSON_COLOR, 2)
                    cv2.putText(img_axes,
                        #text= f'person{count}, {round(float(conf), 3)}',
                        text= f'person',
                        org=(int(xmin), int(ymin-6)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=PERSON_COLOR,
                        thickness=2,
                        lineType=cv2.LINE_4)
                    

        # 人の座標を地図上に描画
        draw_map(people_bottom_w_xyz_list, MAP_WIDTH, MAP_HEIGHT, MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX)

        total_time = rtime.start_and_accumulate(any_in_area)
        img_axes = es.draw_area(img_axes, *WHERE_AREA[0], *WHERE_AREA[1], 0, any_in_area, count_people, total_time)


        img_axes = draw(img_axes,corners12,imgpts)
        #frame1 = cv2.resize(frame1,dsize=(frame1.shape[1]*2,frame1.shape[0]*2))
        if ret:
            img_axes = es.line_update(img_axes)
            cv2.imshow('camera1', img_axes)      #カメラの画像の出力

        """big_img = img_axes.copy()
        big_img = cv2.resize(big_img,dsize=(big_img.shape[1]*3//2,big_img.shape[0]*3//2))       # これで出力画像のサイズを変更
        cv2.imshow('iSpace_big_img', big_img)      #カメラの画像の出力"""


        #繰り返し分から抜けるためのif文
        key =cv2.waitKey(1)
        if key == 27:   #Escで終了
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()


"""
【参考】
カメラキャリブレーション
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
姿勢推定
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_calib3d/py_pose/py_pose.html#pose-estimation
https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
画像座標系からカメラ座標系への変換
https://mem-archive.com/2018/10/13/post-682/
直線同士の最接近点
https://phst.hateblo.jp/entry/2020/02/29/000000
"""