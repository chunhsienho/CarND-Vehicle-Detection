from collections import deque
from Lane_finder import LaneFinder
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from vehicle_detect import find_cars
import cv2
import numpy as np

# vehicle detection constants
searches = [
    (380, 500, 1.0, 1, (0, 0, 255)),  # 64x64
    (400, 600, 1.587, 2, (0, 255, 0)),  # 101x101
    (400, 710, 2.52, 2, (255, 0, 0)),  # 161x161
    (400, 720, 4.0, 2, (255, 255, 0)),  # 256x256
]
nframes_to_keep = 10
nframes = deque([], nframes_to_keep)
frame_decay = 0.85

# save data across frames
lf = LaneFinder()
nframes_heat = None
heat_zeros = None


def detect_road(undist):  #from def img_pipeline in p4 section1 blog 17
    """ Extract road pixels using undistorted image"""
    # calculate road
    warped = lf.perspective_transform(undist)
    combined_binary = lf.combined_thresholding(warped)
    left_fit, right_fit = lf.find_lines(combined_binary)

    # calculate lane line points
    ploty = np.linspace(0, lf.height-1, lf.height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # filp right points upside down to make a close shape
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cp = np.zeros_like(undist)
    cv2.fillPoly(cp, np.int_([pts]), (0, 255, 0))
    road = lf.perspective_transform(cp, reverse=True)

    # Calculate curvature radius of both lines and average them
    left_rad = lf.left_lines.current_curvature_radius()
    right_rad = lf.right_lines.current_curvature_radius()
    mean_rad = (left_rad + right_rad)/2

    # Calculate center of the road to the center of the image
    left_x = lf.left_lines.best_x()   # left line bottom pixel x position
    right_x = lf.right_lines.best_x()  # right line bottom pixel x position
    offset_x = (1280/2) - (left_x + right_x)/2
    offset_direction = "right" if offset_x > 0 else "left"
    offset_x_meter = offset_x * 3.7/700

    # write radius and offset onto image
    
    
    rad_text = 'Radius of Curvature = %.0f' % (mean_rad)
    offset_text = 'Vehicle is %.2f m %s of center' % (
        abs(offset_x_meter), offset_direction)

    return road, rad_text, offset_text

#################from vehicle detect

def _add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def _apply_threshold(heatmap, threshold):
    result = np.copy(heatmap)
    result[heatmap <= threshold] = 0
    return result


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = (
            (np.min(nonzerox), np.min(nonzeroy)),
            (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def detect_vehicle(rgb_img):
    global nframes_heat
    global heat_zeros
    global frames_decay

    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    bbox_list = []
    for ystart, ystop, scale, cells_per_step, color in searches:
        bboxes = find_cars(img, ystart, ystop, scale, cells_per_step)
        if len(bboxes) > 0:
            bbox_list.append(bboxes)

    # initialize data across frames if None
    if nframes_heat is None:
        nframes_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat_zeros = np.zeros_like(img[:, :, 0]).astype(np.float)

    # calculate single frame heatmap
    one_frame_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    if len(bbox_list) > 0:
        one_frame_heat = _add_heat(one_frame_heat, np.concatenate(bbox_list))

    # substract heat older than nframes
    if len(nframes) == nframes_to_keep:
        oldest_heat = nframes.popleft()
        nframes_heat = (
            nframes_heat -
            oldest_heat * (frame_decay ** (nframes_to_keep - 1)))

    nframes.append(one_frame_heat)
    nframes_heat = nframes_heat * frame_decay + one_frame_heat

    # Apply threshold to help remove false positives
    heat = _apply_threshold(nframes_heat, 20)
    # Visualize the heatmap for video

    # Find final boxes from heatmap using label function
    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(rgb_img), labels)
    return draw_img



def img_pipeline(img):
    undist = lf.undistort(img)
    result = detect_vehicle(undist)

    road, rad_text, offset_text = detect_road(undist)
    result = cv2.addWeighted(result, 1.0, road, 0.3, 0)
    # write radius and offset onto image
    result = cv2.putText(
        result, rad_text,
        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    result = cv2.putText(
        result, offset_text, (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (255, 255, 255), 3)
    return result


# NOTE: f1_image function expects color images!!
outfile = 'combined_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(img_pipeline)
white_clip.write_videofile(outfile, audio=False)
