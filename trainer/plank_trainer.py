import cv2
import math
import numpy as np


def angle_between_points(p1, p2, p3):
    """
    Returns the angle in degrees between three points, with p2 as the vertex
    """
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_product = math.sqrt((v1[0] ** 2 + v1[1] ** 2) * (v2[0] ** 2 + v2[1] ** 2))
    angle = math.degrees(math.acos(dot_product / magnitude_product))
    return angle


def planks_assessment(result: dict, conf_thres: float = 0.5):
    """
    Takes in a single key point and returns if the posture in the current frame is correct or not.
    :param result: dict
    :param conf_thres: Confidence threshold of every key points
    :return: True id posture is correct else False
    """
    # check left hand
    if all(x > conf_thres for x in
           (result['left_shoulder'][2], result['left_shoulder'][2], result['left_shoulder'][2])):
        angle_on_left_hand = angle_between_points(p1=result['left_shoulder'][:2],
                                                  p2=result['left_elbow'][:2],
                                                  p3=result['left_wrist'][:2])
        if angle_on_left_hand < 70:
            return False, f"Incorrect Posture: Left hand Angle: {angle_on_left_hand}"
        else:
            pass

    # check right hand
    if all(x > conf_thres for x in (result['right_shoulder'][2], result['right_shoulder'][2], result['right_shoulder'][2])):
        angle_on_right_hand = angle_between_points(p1=result['right_shoulder'][:2],
                                                   p2=result['right_elbow'][:2],
                                                   p3=result['right_wrist'][:2])
        if angle_on_right_hand < 70:
            return False, f"Incorrect Posture: Right hand angle: {angle_on_right_hand}"
        else:
            pass

    # check left leg
    if all(x > conf_thres for x in (result['left_hip'][2], result['left_knee'][2], result['left_ankle'][2])):
        angle_on_left_leg = angle_between_points(p1=result['left_hip'][:2],
                                                 p2=result['left_knee'][:2],
                                                 p3=result['left_ankle'][:2])
        if angle_on_left_leg < 160:
            return False, f"Incorrect Posture: Left leg angle: {angle_on_left_leg}"
        else:
            pass

    # check right leg
    if all(x > conf_thres for x in (result['right_hip'][2], result['right_knee'][2], result['right_ankle'][2])):
        angle_on_right_leg = angle_between_points(p1=result['right_hip'][:2],
                                                 p2=result['right_knee'][:2],
                                                 p3=result['right_ankle'][:2])
        if angle_on_right_leg < 160:
            return False, f"Incorrect Posture: Right leg angle: {angle_on_right_leg}"
        else:
            pass

    return True, "Good Posture. Hit your core."


def plot_planks_skeleton(image, result, is_correct, response, steps=3):
    if result is not None:
        kpts = result
        # Plot the skeleton and keypoints for coco datatset
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        radius = 5
        num_kpts = len(kpts) // steps

        for kid in range(num_kpts):
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                overlay = image.copy()
                alpha = 0.4
                cv2.circle(overlay, (int(x_coord), int(y_coord)), 8, (int(220), int(237), int(245)), 8)
                cv2.circle(image, (int(x_coord), int(y_coord)), 5, (int(255), int(255), int(255)), -1)
                # im = output
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
            pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
            if steps == 3:
                conf1 = kpts[(sk[0] - 1) * steps + 2]
                conf2 = kpts[(sk[1] - 1) * steps + 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(image, pos1, pos2, (int(255), int(255), int(255)), thickness=2)

    text = response
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    if is_correct:
        color = (205, 51, 51)  # brown3
    else:
        color = (69, 139, 116)  # aquamarine4
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = 10
    y = text_size[1] + 10
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
