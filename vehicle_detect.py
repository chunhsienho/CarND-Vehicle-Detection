from Train_classifier import convertors
from Train_classifier import get_hog_features
from Train_classifier import color_hist
from Train_classifier import bin_spatial
import cv2
import numpy as np
import pickle
print('vehicle_detection')

svc = None
X_scaler = None
pix_per_cell = None
spatial_size = None
hist_bins = None
color_space = None

with open('./svc_pickle.p', 'rb') as f:
    data_pickle = pickle.load(f)
    svc = data_pickle['svc']
    X_scaler = data_pickle['scaler']
    pix_per_cell = data_pickle['pix_per_cell']
    spatial_size = data_pickle['spatial_size']
    hist_bins = data_pickle['hist_bins']
    color_space = data_pickle['color_space']


def find_cars(img, ystart, ystop, scale, cells_per_step):

    car_windows = []
    # ignore upper half image (sky, trees ... etc)
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, convertors[color_space])
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(
            ctrans_tosearch,
            (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1)
    hog2 = get_hog_features(ch2)
    hog3 = get_hog_features(ch3)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[
                ypos:ypos+nblocks_per_window,
                xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[
                ypos:ypos+nblocks_per_window,
                xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[
                ypos:ypos+nblocks_per_window,
                xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop+window, xleft:xleft+window],
                (64, 64))

            # Get color features
            hist_features = color_hist(subimg, nbins=hist_bins)
            spatial_features = bin_spatial(subimg, size=spatial_size)

            # Scale features and make a prediction
            all_features = np.hstack(
                (hist_features, spatial_features, hog_features)).reshape(1, -1)

            test_features = X_scaler.transform(all_features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                # confidence = svc.decision_function(test_features)
                # print(confidence)
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                car_windows.append((
                    (xbox_left, ytop_draw+ystart),
                    (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

    return car_windows
