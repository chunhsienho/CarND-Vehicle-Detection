# -*- coding: utf-8 -*-
"""
Calculates the camera calibration matrix
"""

from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import os
import pickle

# chessboard images folder, size and number of corners
# note that some calibration images are 720x1281
chessboard_imgs = './camera_cal/*.jpg'
img_size = (720, 1280)
nx, ny = 9, 5

# distortion matrix pickle file
dist_mtx_file = 'camera_matrix.p'


class CameraCalibrator():

    def calibrate_camera(self, force=False):
        """Calculate distortion matrix and save to file"""
        if os.path.isfile(dist_mtx_file):
            self.camera_matrix = pickle.load(open(dist_mtx_file, 'rb'))
        else:
            obj_point = np.zeros((nx*ny, 3), np.float32)
            obj_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
            obj_points = []  # list of 3D obj coordinates
            img_points = []  # list of 2D coordinates found on image plane

            # use tqdm to display progress on screen
            for img_path in tqdm(glob(chessboard_imgs)):
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
                if ret is True:
                    obj_points.append(obj_point)
                    img_points.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points, img_points, img_size, None, None)

            # Save the camera calibration result for later use
            self.camera_matrix = {
                'mtx': mtx,
                'dist': dist,
            }
            # Save camera matrix to avoid recalibrating every time
            # run with force=True to force recalibrate
            pickle.dump(self.camera_matrix, open(dist_mtx_file, 'wb'))

    def undistort_image(self, img):
        if not hasattr(self, 'camera_matrix'):
            self.calibrate_camera()

        mtx = self.camera_matrix['mtx']
        dist = self.camera_matrix['dist']
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        return undistorted

if __name__ == "__main__":

    calibrator = CameraCalibrator()
    calibrator.calibrate_camera(force=False)
