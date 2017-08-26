from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
import glob
import numpy as np
import pickle
import time

convertors = {
    'RGB': cv2.COLOR_BGR2RGB,
    'HLS': cv2.COLOR_BGR2HLS,
    'YUV': cv2.COLOR_BGR2YUV,
    'YCrCb': cv2.COLOR_BGR2YCrCb,
    'Lab': cv2.COLOR_BGR2Lab,
    'Luv': cv2.COLOR_BGR2Luv,
}

scv = None
X_scaler = None
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (16, 16)
hist_bins = 128
color_space = 'YCrCb'

vehicle_imgs = list(glob.glob('./vehicles/**/*.png'))
nonvehicle_imgs = list(glob.glob('./non-vehicles/**/*.png'))


def get_hog_features(img, vis=False, feature_vec=False):
    return hog(
        img, orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        visualise=vis, feature_vector=feature_vec)


def color_hist(img, nbins=128, bins_range=(0, 256), visualize=False):
    channel1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    features = np.concatenate((channel1[0], channel2[0], channel3[0]))
    if visualize:
        print(features.shape)
        return features, (channel1, channel2, channel3)
    else:
        return features


def bin_spatial(img, size=(16, 16)):
    resize_img = cv2.resize(img, size)
    color1 = resize_img[:, :, 0].ravel()
    color2 = resize_img[:, :, 1].ravel()
    color3 = resize_img[:, :, 2].ravel()
    features = np.hstack((color1, color2, color3))
    return features


def extract_features(img, cspace='HLS'):
    img = cv2.imread(img)
    convertor = convertors[cspace]
    cvt_img = cv2.cvtColor(img, convertor)

    hist_features = color_hist(cvt_img, nbins=hist_bins)
    bin_features = bin_spatial(cvt_img, size=spatial_size)

    channel1 = get_hog_features(cvt_img[:, :, 0]).ravel()
    channel2 = get_hog_features(cvt_img[:, :, 1]).ravel()
    channel3 = get_hog_features(cvt_img[:, :, 2]).ravel()

    return np.concatenate((
        hist_features, bin_features, channel1, channel2, channel3))


def main():
    t0 = time.time()
    car_features = np.array(
        [extract_features(img, cspace=color_space) for img in vehicle_imgs])
    noncar_features = np.array(
        [extract_features(img, cspace=color_space) for img in nonvehicle_imgs])
    t1 = time.time()
    extract_duration = round(t1 - t0, 2)

# define features
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
# define labels
    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(noncar_features))))

    rand_state = np.random.randint(0, 101)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()
    t2 = time.time()
    svc.fit(X_train, y_train)
    t3 = time.time()

    train_duration = round(t3-t2, 2)
    t4 = time.time()
    accuracy = svc.score(X_test, y_test)
    t5 = time.time()
    score_duration = round(t5-t4, 2)

    print('| Color space | Feature extraction |'
          'Training time | Predict Time | Accuract |')
    print('|-------|-------|-------|-------|-------|')
    print('| %s | %.2f | %.2f | %.2f | %.4f |' % (
        color_space, extract_duration, train_duration,
        score_duration, accuracy))

    data_pickle = {
        'svc': svc,
        'scaler': X_scaler,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'spatial_size': spatial_size,
        'hist_bins': hist_bins,
        'color_space': color_space,
    }

    filename = 'svc_pickle.p'
    with open(filename, 'wb') as f:
        pickle.dump(data_pickle, f)
        print('params saved to %s' % filename)


if __name__ == '__main__':
    main()
