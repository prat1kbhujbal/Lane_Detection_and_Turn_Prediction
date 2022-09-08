import cv2
import argparse
import numpy as pb
import warnings


def warp(img):
    """Function to transfoem image perpective

    Args:
        img : Input Image

    Returns:
        : Transformed Image and invP matrix
    """
    # Source points
    src = pb.float32([
        (530, 450),
        (760, 450),
        (1125, 660),
        (112, 660)
    ])
    # Output points
    otp = pb.float32([
        (0, 0),
        (1279, 0),
        (1279, 719),
        (0, 719)
    ])

    P = cv2.getPerspectiveTransform(src, otp)
    invP = cv2.getPerspectiveTransform(otp, src)
    warp_img = cv2.warpPerspective(
        img, P, (img.shape[1],
                 img.shape[0]),
        flags=cv2.INTER_LINEAR)

    return warp_img, invP


def curvature(left_ptns, right_ptns):
    """Calculated left and right lane lines curvature

    Args:
        left_ptns : Left line points
        right_ptns : Right line points

    Returns:
        : Left and Right Curvature
    """
    leftx, lefty = left_ptns[:, 0], left_ptns[:, 1]
    rightx, righty = right_ptns[:, 0], left_ptns[:, 1]
    max_y = pb.max(lefty)

    # Pixel values in meters for lane width and dashed line
    dashed_m = 3 / 140
    lane_m = 3.7 / 960

    coeff_y = pb.polyfit(lefty * dashed_m, leftx * (
        lane_m), 2)
    coeff_w = pb.polyfit(righty * dashed_m, rightx * (
        lane_m), 2)

    # Left and Right line curvature
    left_curvem = ((1 +
                    (2 * coeff_y[0] * max_y * dashed_m + coeff_y
                     [1]) ** 2) ** 1.5) / pb.absolute(2 * coeff_y[0])
    right_curvem = ((1 + (2 * coeff_w[
                    0] * max_y * dashed_m + coeff_w[
        1])**2)**1.5) / pb.absolute(2 * coeff_w[0])
    return left_curvem, right_curvem


def polynomial_fit(y_, coeffy, coeffw):
    """2d Polynomial fit
    """
    yellow_l = coeffy[0] * y_**2 + coeffy[1] * y_ + coeffy[2]
    white_l = coeffw[0] * y_**2 + coeffw[1] * y_ + coeffw[2]
    return yellow_l, white_l


def lane_detect(img, yellow_mask, white_mask):
    """Fucntion to fit and draw lane lines using masked lane lines

    Args:
        img : Input image
        yellow_mask : yellow mask
        white_mask : white mask
    """
    warnings.simplefilter('ignore', pb.RankWarning)

    # Find white pixels in yellow mask
    indxx_y, indxy_y = pb.where(yellow_mask == [255])
    coeff_y = pb.polyfit(indxx_y, indxy_y, 2)

    # Find white pixels in white mask
    indxx_w, indxy_w = pb.where(white_mask == [255])
    coeff_w = pb.polyfit(indxx_w, indxy_w, 2)

    y_ = pb.linspace(0, img.shape[0] - 1, img.shape[0])
    yellow_pts, white_pts = polynomial_fit(y_, coeff_y, coeff_w)

    yellowlane_line = (pb.asarray([yellow_pts, y_]).T).astype(pb.int32)
    whitelane_line = (pb.asarray([white_pts, y_]).T).astype(pb.int32)

    # Find center line
    center_line = pb.empty_like(yellowlane_line)
    center_line[:, 0] = (yellow_pts + white_pts) / 2
    center_line[:, 1] = y_
    x_center = pb.array([center_line[0][0], center_line[710][0]])
    y_center = pb.array([center_line[0][1], center_line[710][1]])
    m_center, c = pb.polyfit(x_center, y_center, 1)

    # Turn prediction
    if m_center > 0:
        turn_prediction = "Left Turn"
    elif m_center < 0:
        turn_prediction = "Right Turn"
    else:
        turn_prediction = "Straight"
    return yellowlane_line, whitelane_line, center_line, turn_prediction


def plot(img, yellowlane_line, whitelane_line, center_line):
    """Plot polygon between two lanes and draw arrow on center line
    """
    whitelane_line = cv2.flip(whitelane_line, 0)
    result = pb.vstack((yellowlane_line, whitelane_line))
    lane_region = pb.zeros(
        (img.shape[0],
         img.shape[1],
         img.shape[2]),
        dtype=pb.uint8)
    lanes = pb.zeros(
        (img.shape[0],
         img.shape[1],
         img.shape[2]),
        dtype=pb.uint8)
    # Draw polygon
    cv2.fillPoly(lane_region, [result], (0, 0, 255))
    cv2.polylines(
        lanes, [yellowlane_line],
        False, (0, 255, 255),
        20)
    # Draw lanes
    cv2.polylines(
        lanes, [whitelane_line],
        False, (255, 0, 0),
        20)
    sx, sy, ex, ey = center_line[300][0], center_line[300][1], center_line[400][0], center_line[400][1]
    cv2.arrowedLine(
        lane_region, (ex, ey),
        (sx, sy),
        (0, 255, 255), 20, tipLength=1.5)
    cv2.polylines(
        lane_region, [center_line[400: 650]],
        False, (0, 255, 255),
        20)
    return lane_region, lanes


def threshold(img):
    """Generate yellow and white mask

    Args:
        img: Input image

    Returns:
        : Mask Images
    """
    # image = cv2.flip(image, 1)
    img_copy = img.copy()

    # HSV
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    hsv_low_yellow = pb.array((15, 130, 150))
    hsv_high_yellow = pb.array((30, 255, 255))
    hsv_y = cv2.inRange(hsv, hsv_low_yellow, hsv_high_yellow)

    # LAB
    lab = cv2.cvtColor(img_copy, cv2.COLOR_BGR2LAB)
    l1, a, b = cv2.split(lab)
    maxl1, maxb, meanb = pb.max(l1), pb.max(b), pb.mean(b)
    l1max_yellow = max(80, int(maxl1 * 0.45))
    bmax_yellow = max(int(maxb * 0.70), int(meanb * 1.2))
    labyellow_lower = pb.array((l1max_yellow, 120, bmax_yellow))
    labyellow_upper = pb.array((255, 145, 255))
    lab_y = cv2.inRange(lab, labyellow_lower, labyellow_upper)

    # HLS
    hls = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HLS)
    hlsyellow_lower = pb.array((13, 0, 110))
    hlsyellow_upper = pb.array((24, 200, 255))
    hls_y = cv2.inRange(hls, hlsyellow_lower, hlsyellow_upper)

    hlswhite_lower = pb.array((0, 212, 0))
    hlswhite_upper = pb.array((255, 255, 255))
    hls_w = cv2.inRange(hls, hlswhite_lower, hlswhite_upper)

    # bitwise or all the channels
    masked_yellow = cv2.bitwise_or(hls_y, cv2.bitwise_or(hsv_y, lab_y))
    kernel = (100, 100)
    masked_white = cv2.morphologyEx(hls_w, cv2.MORPH_OPEN, kernel)

    output_white = cv2.bitwise_and(img_copy, img_copy, mask=masked_white)
    output_yellow = cv2.bitwise_and(img_copy, img_copy, mask=masked_yellow)
    output = cv2.bitwise_or(output_white, output_yellow)
    return masked_yellow, masked_white, output


def output_frame(current_frame,
                 warp_image,
                 merged_lanelines,
                 lanes_fit,
                 final_image,
                 left_curv,
                 right_curv,
                 turn_prediction):
    """Resize and put text on images and concatenate to get final frame

    Returns:
        : Final Output Image
    """
    final_image = cv2.putText(
        final_image, "Prediction: " + turn_prediction, (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        1, cv2.LINE_AA)
    current_frame = cv2.resize(current_frame, (480, 240), cv2.INTER_CUBIC)
    current_frame = cv2.putText(
        current_frame, '1.', (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)
    warp_image = cv2.resize(warp_image, (480, 240), cv2.INTER_CUBIC)
    warp_image = cv2.putText(
        warp_image, '2.', (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)
    merged_lanelines = cv2.resize(
        merged_lanelines, (480, 240),
        cv2.INTER_CUBIC)
    merged_lanelines = cv2.putText(
        merged_lanelines, '3.', (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)
    lanes_fit = cv2.resize(lanes_fit, (480, 240), cv2.INTER_CUBIC)
    lanes_fit = cv2.putText(
        lanes_fit, '4.', (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)

    # Vertical Stack
    v_stk = cv2.vconcat(
        [current_frame, warp_image, merged_lanelines, lanes_fit])
    text_img = pb.zeros((240, 1280, 3), dtype=pb.uint8)
    text_img = cv2.putText(
        text_img, "Left Curvature: " + str(round(left_curv, 2)) + " m, " +
        "Right_Curvature: " + str(round(right_curv, 2)) + " m",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)
    text_img = cv2.putText(
        text_img, "Average Curvature: " + str(round(((left_curv + right_curv) / 2), 2)) +
        " m, ", (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)
    text_img = cv2.putText(
        text_img, "Frames : ", (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)
    text_img = cv2.putText(
        text_img,
        "1. Input Image ; 2. Warped Image ; 3. Detected Lane Lines ; 4. Polynomial Fitting",
        (30, 165),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
        2, cv2.LINE_AA)

    # Horizontal Stack
    h_stk = cv2.vconcat([final_image, text_img])
    h_stk = cv2.hconcat([h_stk, v_stk])
    return h_stk


def main():
    '''Main Function'''
    # Arguments
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--FilePath', default='../data_files/challenge.mp4',
        help='Video file path')
    parse.add_argument(
        "--visualize",
        default=True,
        choices=('True', 'False'),
        help="Shows visualization. Default: False.")
    parse.add_argument(
        "--record",
        default=False,
        choices=('True', 'False'),
        help="Records video (../turn_predict.mp4). Default: False.")
    Args = parse.parse_args()
    file_path = Args.FilePath
    visualize = str(Args.visualize)
    video_write = str(Args.record)
    fps = 25
    if video_write == str(True):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            "../turn_predict.mp4", fourcc, fps, (1760, 960))
        print("Writing to Video...")
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't retrieve frame - stream may have ended. Exiting..")
            cap.release()
            cv2.destroyAllWindows()
            break
        warp_img, invP = warp(frame)
        masked_yellow, masked_white, merged = threshold(warp_img)
        yellowlane_line, whitelane_line, center_line, turn_prediction = lane_detect(
            warp_img, masked_yellow, masked_white)
        left_curv, right_curv = curvature(yellowlane_line, whitelane_line)
        average_curv = (left_curv + right_curv) / 2
        lanes_img, lanes_fit = plot(
            warp_img, yellowlane_line, whitelane_line, center_line)
        warp_inv = cv2.warpPerspective(
            lanes_img, invP, (frame.shape[1],
                              frame.shape[0]),
            flags=cv2.INTER_LINEAR)
        final_image = cv2.addWeighted(frame, 0.9, warp_inv, 0.7, 0)
        output = output_frame(frame,
                              warp_img,
                              merged,
                              lanes_fit,
                              final_image,
                              left_curv,
                              right_curv,
                              turn_prediction)
        if video_write == str(True):
            out.write(output)
        if visualize == str(True):
            cv2.imshow("Lane Prediction and Curvature", output)
            cv2.waitKey(fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if video_write == str(True):
        out.release()


if __name__ == '__main__':
    main()
