import numpy as np
import cv2
from collections import deque
from camera_calibration import CameraCalibrator


class Line():
    """class to receive the characteristics of each line detection"""
    def __init__(self, frames_to_keep=5):
        self.recent_nx = deque([], frames_to_keep)
        self.recent_nfits = deque([], frames_to_keep)
        self.recent_curvrad = deque([], frames_to_keep)
        self.recent_x_pos = deque([], frames_to_keep)
        self.last_found = False

    def found_fits(self):
        return [f for f in self.recent_nfits if f is not None]

    def best_fit(self):
        """Returns the best fit according to the last n fits"""
        found_fits = self.found_fits()
        if not found_fits:
            return None
        else:
            average_fit = np.mean(found_fits, axis=0)
            return average_fit

    def best_x(self):
        """Returns the average non None x positions"""
        found_x_pos = [x for x in self.recent_x_pos if x is not None]
        return np.mean(found_x_pos)

    def line_valid(self, line_fit):
        """Decides whether the new appended line_fit is a valid line"""
        curvrad = self.curvature_radius(line_fit)
        xpos = self.calc_x_pos(line_fit)
        best_x = self.best_x()

        # Regard curvature too low or large x difference as invalid
        if curvrad < 350.0:
            return False
        if best_x and abs(xpos - best_x) > 70:
            return False
        return True

    def append_fit(self, line_fit):
        """Record new fitted line after sanity check"""
        if len(self.recent_nfits) > 0 and not self.line_valid(line_fit):
            if self.last_found:
                self.recent_nfits.append(None)
                self.last_found = False
        else:
            self.last_found = True
            self.recent_nfits.append(line_fit)
            self.recent_x_pos.append(self.calc_x_pos(line_fit))

    def curvature_radius(self, line_fit):
        """Calculate the curvature of fitted line"""
        ym_per_pix = 30/720
        xm_per_pix = 3.7/700
        # refit line in world space
        ploty = np.array([100, 200, 300, 400, 500])
        # ploty = np.linspace(0, self.height-1, self.height)
        plotx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
        fit_cr = np.polyfit(ploty*ym_per_pix, plotx*xm_per_pix, 2)

        y_eval = 720*ym_per_pix
        curvrad = (
            ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) /
            np.absolute(2*fit_cr[0]))
        return curvrad

    def current_curvature_radius(self):
        """Calculate the curvature radius using best fit"""
        best_fit = self.best_fit()
        return self.curvature_radius(best_fit)

    def calc_x_pos(self, line_fit):
        """Gets the bottom line pixel x coordinate"""
        y_eval = 720
        return np.int(line_fit[0]*y_eval**2 + line_fit[1]*y_eval + line_fit[2])


class LaneFinder():
    """Finds laneline inside car camera images"""
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.left_lines = Line()
        self.right_lines = Line()
        self.s_magnitude_thresh = (175, 255)
        self.sobel_kernel = 7
        self.m_thresh = (14, 255)
        self.d_thresh = (0.0, 0.73)
        self.camera_calibrator = CameraCalibrator()

    def undistort(self, img):
        """Undistort image using camera matrix"""
        return self.camera_calibrator.undistort_image(img)

    def s_magnitude(self, img):
        """Returns magnitude thresholded binary image of
        the S channel in HLS color space
        """
        thresh = self.s_magnitude_thresh
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        magnitude_binary = np.zeros_like(s_channel)
        magnitude_binary[
            (s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return magnitude_binary

    def l_direction(self, img):
        """ Apply sobel filter on L(HLS) channel and threshold,
        filter pixels direction inside self.d_thresh and
        filter pixels magnitude inside self.m_thresh
        """
        sobel_kernel = self.sobel_kernel
        m_thresh = self.m_thresh
        d_thresh = self.d_thresh

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        magnitude = np.sqrt(np.square(sobelx)+np.square(sobely))
        scaled = np.uint8(255*magnitude/np.max(magnitude))
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        direction = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(direction)
        binary_output[
            (direction >= d_thresh[0]) &
            (direction <= d_thresh[1]) &
            (scaled >= m_thresh[0]) &
            (scaled <= m_thresh[1])] = 1
        return binary_output

    def combined_thresholding(self, img):
        """Returns the combined result of all thresholdings"""
        s_mag = self.s_magnitude(img)
        l_dir = self.l_direction(img)
        combined_binary = np.zeros_like(img[:, :, 1])
        combined_binary[(s_mag == 1) | (l_dir == 1)] = 1
        return combined_binary

    def get_perspective_transform_matrix(self, reverse=False):
        matrix_key = 'perspective_transform_mtx'
        if reverse:
            matrix_key = 'reverse_perspective_transform_mtx'

        matrix = getattr(self, matrix_key, None)
        if matrix is not None:
            return matrix

        # no previous stored matrix, calculate one
        tls = (563, 470)  # top left source point
        bls = (220, 700)  # bottom left source point
        tld = (300, 300)  # top left destination
        bld = (300, 720)  # bottom left destination

        src = np.float32([
            [tls[0], tls[1]],
            [self.width-tls[0], tls[1]],
            [self.width-bls[0], bls[1]],
            [bls[0], bls[1]]
        ])

        dst = np.float32([
            [tld[0], tld[1]],
            [self.width-tld[0], tld[1]],
            [self.width-tld[0], bld[1]],
            [bld[0], bld[1]],
        ])

        if reverse:
            transform_mtx = cv2.getPerspectiveTransform(dst, src)
        else:
            transform_mtx = cv2.getPerspectiveTransform(src, dst)

        # save matrix for later use
        setattr(self, matrix_key, transform_mtx)
        return transform_mtx

    def perspective_transform(self, img, reverse=False):
        """Transform car camera image into birds eye view"""
        transform_mtx = self.get_perspective_transform_matrix(reverse=reverse)
        shape = (self.width, self.height)
        warped = cv2.warpPerspective(
            img, transform_mtx, shape, flags=cv2.INTER_LINEAR)
        return warped

    def histogram_find_lines(self, binary_warped):
        """Find left/right lane line indices from the binary warped
        image without knowing previous line positions using the
        hostogram/sliding window method
        """
        width, height = self.width, self.height
        nwindows = 9
        window_height = np.int(height/nwindows)

        histogram = np.sum(binary_warped[int(height/2):, :], axis=0)
        midpoint = np.int(width/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []
        windows = []  # record search window for visualization

        margin = 100  # half window size
        minpix = 100   # least pixels to be recognized as found

        for window in range(nwindows):
            win_y_low = height - (window+1)*window_height
            win_y_high = height - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            windows.append(
                (win_xleft_low, win_y_low, win_xleft_high, win_y_high))
            windows.append(
                (win_xright_low, win_y_low, win_xright_high, win_y_high))

            good_left_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &
                (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &
                (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def convolution_find_lines(self, binary_warped, left_fit, right_fit):
        """Find line around known previous lines
        left_fit, right_fit: previously fitted left, right lines
        """
        window_width = 50
        hww = 25  # half window width
        n_windows = 9
        window_height = self.height/n_windows
        window_centroids = []
        margin = 100  # How much to slide left/right for searching
        window = np.ones(window_width)
        offset = np.int((window_width + margin)/2)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = []
        right_lane_inds = []

        for level in range(n_windows):
            y_min = np.int(level*window_height)
            y_max = np.int(y_min + window_height)
            layer = binary_warped[y_min:y_max, :]
            image_layer = np.sum(layer, axis=0)
            conv_signal = np.convolve(window, image_layer)

            y_eval = (y_min + y_max)/2
            l_base = np.int(
                left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2])
            l_base = max(0, min(self.width, l_base))
            l_center = (
                np.argmax(conv_signal[
                    max(0, l_base-offset): min(self.width, l_base+offset)]) + l_base - offset - hww)
            r_base = np.int(right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2])
            r_base = max(0, min(self.width, r_base))
            r_center = (np.argmax(conv_signal[max(0, r_base-offset): min(self.width, r_base+offset)]) +r_base - offset - hww)
            window_centroids.append((l_center, r_center))
            good_left_inds = (
                (nonzeroy >= y_min) &
                (nonzeroy <= y_max) &
                (nonzerox >= (l_center-hww)) &
                (nonzerox <= (l_center+hww))).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= y_min) &
                (nonzeroy <= y_max) &
                (nonzerox >= (r_center-hww)) &
                (nonzerox <= (r_center+hww))).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def find_lines(self, img):
        """Find lane line"""
        prev_left_fit = self.left_lines.best_fit()
        prev_right_fit = self.right_lines.best_fit()
        if prev_left_fit is None or prev_right_fit is None:
            naive_left_fit, naive_right_fit = self.histogram_find_lines(img)
            if prev_left_fit is None:
                prev_left_fit = naive_left_fit
            if prev_right_fit is None:
                prev_right_fit = naive_right_fit

        new_left_fit, new_right_fit = self.convolution_find_lines(img, prev_left_fit, prev_right_fit)
        
        # TODO(Olala): sanity check before append
        self.left_lines.append_fit(new_left_fit)
        self.right_lines.append_fit(new_right_fit)
        avg_left_fit = self.left_lines.best_fit()
        avg_right_fit = self.right_lines.best_fit()
        assert avg_left_fit is not None
        assert avg_right_fit is not None
        return avg_left_fit, avg_right_fit
