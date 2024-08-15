from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import numpy as np
from stqdm import stqdm
# STEP 1: Import the necessary modules.
import sys

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    annotatedimage=annotated_image
    annotated_image.release()
    return annotatedimage


def draw_landmarks_and_floatedness_on_image(
    rgb_image, detection_result, frame, floating_handinfos
):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        if [frame, handedness[0].category_name] in floating_handinfos:
            floatedness = "floating"
        else:
            floatedness = "notfloating"

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name},{floatedness}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    annotatedimage=annotated_image
    return annotatedimage


def draw_keyboard_on_image(rgb_image, keylist):
    annotated_image = np.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(keylist)):
        key_landmarks = keylist[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=point[0], y=point[1], z=0)
                for point in key_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(annotated_image, hand_landmarks_proto)

    return annotated_image


sys.path.append("..")


class handclass:
    def __init__(self, handtype, handlandmark, handframe):
        self.handtype = handtype
        self.handlandmark = handlandmark
        self.handframe = handframe


class distanceclass:
    def __init__(self, handtype, distancesum, handframe):
        self.handtype = handtype
        self.distancesum = distancesum
        self.handframe = handframe


def landmarkdistance(landmarka, landmarkb):
    return ((landmarka.x - landmarkb.x) ** 2 + (landmarka.y - landmarkb.y) ** 2) ** 0.5


def avgcalc(handlist):
    handdistancelist = []

    for handsinfo in handlist:
        handdistanceinfo = []
        for hand in handsinfo:
            distancesum = 0
            for i in [1, 5, 9, 13, 17]:  # OR, for i in range(21):
                distancesum += landmarkdistance(
                    hand.handlandmark[0], hand.handlandmark[i]
                )
                # distancesum += 0.5*landmarkdistance(hand.handlandmark[5],hand.handlandmark[17])
            handdistanceinfo.append(
                distanceclass(
                    handtype=hand.handtype,
                    distancesum=distancesum,
                    handframe=hand.handframe,
                )
            )
        handdistancelist.append(handdistanceinfo)
    avg = 0
    avgcounter = 0
    for i in handdistancelist:
        for hand in i:
            avg += hand.distancesum
            avgcounter += 1

    avg = avg / avgcounter

    lowavg = 0
    lowavgcounter = 0
    for i in handdistancelist:
        for hand in i:
            if hand.distancesum <= avg:
                lowavg += hand.distancesum
                lowavgcounter += 1
    lowavg = lowavg / lowavgcounter
    return handdistancelist, lowavg


def faultyframes(handlist):
    faultyframes = []
    for handsinfo in handlist:
        if len(handsinfo) >= 3:
            faultyframes.append(handsinfo[0].handframe)
        else:
            temphandlist = []
            for hand in handsinfo:
                if hand.handtype in temphandlist:
                    faultyframes.append(hand.handframe)
                temphandlist.append(hand.handtype)

    return faultyframes


def mymetric(
    handdistancelist, lowavg, handtype, frame, frame_count, a, b, c, faultyframes
):
    framerange = [*range(int(max(0, frame - a)), int(min(frame + b, frame_count)))]
    l = len(framerange)
    tempframes = []
    availableframes = []
    templist = []
    thehand = None
    value = 0
    for handdistanceinfo in handdistancelist:
        for hand in handdistanceinfo:
            if hand.handframe in framerange:
                if hand.handtype == handtype:
                    tempframes.append(hand.handframe)
            if hand.handframe == frame:
                if hand.handtype == handtype:
                    thehand = hand  ## 코드 속도를 위해 수정해야할듯, 이부분은 따로 뺄 필요가 있음
    for frames in tempframes:
        if frames not in faultyframes:
            availableframes.append(frames)

    for handdistanceinfo in handdistancelist:
        for hand in handdistanceinfo:
            templist = [
                *range(int(max(0, frame - a)), int(min(frame + b, frame_count)))
            ]
            if (
                hand.handframe in framerange
            ):  ## 여기의 hand들은 전부 framerange 안에 있음.
                if hand.handtype == handtype:
                    for availableframe in availableframes:
                        if availableframe < frame:
                            value += hand.distancesum * (a - frame + availableframe) / a
                            templist.remove(availableframe)
                        else:
                            value += hand.distancesum * (b + frame - availableframe) / b
                            templist.remove(availableframe)
                    for leftframe in templist:
                        if leftframe < frame:
                            value += thehand.distancesum * (a - frame + leftframe) / a
                        else:
                            value += thehand.distancesum * (b + frame - leftframe) / b

    value = value / l
    return ((c * thehand.distancesum) + (1 - c) * value) - 0.3 * lowavg

def detectfloatingframes(handlist, frame_count, faultyframes):
    metriclist = []
    metricavg = 0
    metricavgcounter = 0
    floatingframes = []
    handdistancelist, lowavg = avgcalc(handlist)
    index=0
    for _ in stqdm(range(len(handlist)), desc="Detecting floating hands..."):
        handsinfo = handlist[index]
        for hand in handsinfo:
            metriclist.append(
                [
                    hand.handframe,
                    hand.handtype,
                    mymetric(
                        handdistancelist,
                        lowavg,
                        hand.handtype,
                        hand.handframe,
                        frame_count,
                        7,
                        7,
                        0.85,
                        faultyframes,
                    ),
                ]
            )
        index+=1

    for metric in metriclist:
        metricavg += metric[2]
        metricavgcounter += 1
    metricavg = metricavg / metricavgcounter

    bighandcounter = 0
    for metric in metriclist:
        if metric[2] > metricavg * 1.4:   #bighand: 영상에 포함된 앞뒤의 매우 큰 손 부분을 잘라주기 위함.
            bighandcounter += 1
    for metric in metriclist:
        if (
            metric[2] > metricavg * (1.3 - 1.3 * bighandcounter / len(metriclist))
        ):  # Differs from how many and how intensive of floating occurs in the whole playing
            floatingframes.append([metric[0], metric[1]])

    #영상 길이가 짧으면 결과가 이상하게 나와서
    #floatingframes=[]

    return floatingframes


def distortioncoefficient(i, distortion):
    if i < 26:
        return distortion * (i - 26) / abs(i - 26) * (-((abs(i - 26) - 13) ** 2) + 169)
    elif i == 26:
        return 0
    if i > 26:
        return (
            0.3
            * distortion
            * (i - 26)
            / abs(i - 26)
            * (-((abs(i - 26) - 13) ** 2) + 169)
        )  ## Fits in our (MACLAB@KAIST) recording environment.


def generatekeyboard(
    lu, ru, ld, rd, blackratio, distortion=0,
):  # 05/12 update: 검은 건반을 가운데가 아니라 실제 건반에 맞게 위치 조정
    # 05/14 update: 카메라 각도에 맞게 검은 건반 위치 미세 조정
    # 07/02 update: x축 distortion 보정

    idx_black = [n for n in range(1, 52) if ((n) % 7) in [0, 1, 3, 4, 6]]

    bottompoints = [
        [
            ld[0] * (52 - i + distortioncoefficient(i, distortion)) / 52
            + rd[0] * (i + distortioncoefficient(i, distortion)) / 52,
            ld[1] * (52 - i) / 52 + rd[1] * i / 52,
        ]
        for i in range(53)
    ]

    toppoints = [
        [
            lu[0] * (52 - i + distortioncoefficient(i, distortion)) / 52
            + ru[0] * (i + distortioncoefficient(i, distortion)) / 52,
            lu[1] * (52 - i) / 52 + ru[1] * i / 52,
        ]
        for i in range(53)
    ]

    topblackpoints = [
        [
            [
                lu[0] * (52 - i + 1 / 4 + distortioncoefficient(i, distortion)) / 52
                + ru[0] * (i - 1 / 4 + distortioncoefficient(i, distortion)) / 52,
                lu[1] * (52 - i + 1 / 4) / 52 + ru[1] * (i - 1 / 4) / 52,
            ],
            [
                lu[0] * (52 - i - 1 / 4 + distortioncoefficient(i, distortion)) / 52
                + ru[0] * (i + 1 / 4 + distortioncoefficient(i, distortion)) / 52,
                lu[1] * (52 - i - 1 / 4) / 52 + ru[1] * (i + 1 / 4) / 52,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i - 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                lu[1] * (52 - i + 1 / 4 + 1 / 12) / 52
                + ru[1] * (i - 1 / 4 - 1 / 12) / 52,
            ],
            [
                lu[0]
                * (52 - i - 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i + 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                lu[1] * (52 - i - 1 / 4 + 1 / 12) / 52
                + ru[1] * (i + 1 / 4 - 1 / 12) / 52,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i - 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                lu[1] * (52 - i + 1 / 4 - 1 / 12) / 52
                + ru[1] * (i - 1 / 4 + 1 / 12) / 52,
            ],
            [
                lu[0]
                * (52 - i - 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i + 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                lu[1] * (52 - i - 1 / 4 - 1 / 12) / 52
                + ru[1] * (i + 1 / 4 + 1 / 12) / 52,
            ],
        ]
        for i in idx_black
    ]

    bottomblackpoints = [
        [
            [
                lu[0] * (52 - i + 1 / 4 + distortioncoefficient(i, distortion)) / 52
                + ru[0] * (i - 1 / 4 + distortioncoefficient(i, distortion)) / 52,
                (lu[1] * (52 - i + 1 / 4) / 52 + ru[1] * (i - 1 / 4) / 52)
                * (1 - blackratio)
                + (ld[1] * (52 - i + 1 / 4) / 52 + rd[1] * (i - 1 / 4) / 52)
                * blackratio,
            ],
            [
                lu[0] * (52 - i + distortioncoefficient(i, distortion)) / 52
                + ru[0] * (i + distortioncoefficient(i, distortion)) / 52,
                (lu[1] * (52 - i) / 52 + ru[1] * (i) / 52) * (1 - blackratio)
                + (ld[1] * (52 - i) / 52 + rd[1] * (i) / 52) * blackratio,
            ],
            [
                lu[0] * (52 - i - 1 / 4 + distortioncoefficient(i, distortion)) / 52
                + ru[0] * (i + 1 / 4 + distortioncoefficient(i, distortion)) / 52,
                (lu[1] * (52 - i - 1 / 4) / 52 + ru[1] * (i + 1 / 4) / 52)
                * (1 - blackratio)
                + (ld[1] * (52 - i - 1 / 4) / 52 + rd[1] * (i + 1 / 4) / 52)
                * blackratio,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i - 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                (
                    lu[1] * (52 - i + 1 / 4 + 1 / 12) / 52
                    + ru[1] * (i - 1 / 4 - 1 / 12) / 52
                )
                * (1 - blackratio)
                + (
                    ld[1] * (52 - i + 1 / 4 + 1 / 12) / 52
                    + rd[1] * (i - 1 / 4 - 1 / 12) / 52
                )
                * blackratio,
            ],
            [
                lu[0]
                * (52 - i - 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i + 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                (
                    lu[1] * (52 - i - 1 / 4 + 1 / 12) / 52
                    + ru[1] * (i + 1 / 4 - 1 / 12) / 52
                )
                * (1 - blackratio)
                + (
                    ld[1] * (52 - i - 1 / 4 + 1 / 12) / 52
                    + rd[1] * (i + 1 / 4 - 1 / 12) / 52
                )
                * blackratio,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i - 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                (
                    lu[1] * (52 - i + 1 / 4 - 1 / 12) / 52
                    + ru[1] * (i - 1 / 4 + 1 / 12) / 52
                )
                * (1 - blackratio)
                + (
                    ld[1] * (52 - i + 1 / 4 - 1 / 12) / 52
                    + rd[1] * (i - 1 / 4 + 1 / 12) / 52
                )
                * blackratio,
            ],
            [
                lu[0]
                * (52 - i - 1 / 4 - 1 / 12 + distortioncoefficient(i, distortion))
                / 52
                + ru[0]
                * (i + 1 / 4 + 1 / 12 + distortioncoefficient(i, distortion))
                / 52,
                (
                    lu[1] * (52 - i - 1 / 4 - 1 / 12) / 52
                    + ru[1] * (i + 1 / 4 + 1 / 12) / 52
                )
                * (1 - blackratio)
                + (
                    ld[1] * (52 - i - 1 / 4 - 1 / 12) / 52
                    + rd[1] * (i + 1 / 4 + 1 / 12) / 52
                )
                * blackratio,
            ],
        ]
        for i in idx_black
    ]

    # topblackpoints: 중심에서 왼쪽으로 1/4, 오른쪽으로 1/4, 왼쪽으로 1/4+1/12, 오른쪽으로 1/4-1/12, 왼쪽으로 1/4-1/12, 오른쪽으로 1/4+1/12 떨어져 있는 점들의 list.
    # bottomplackpoints: 중심에서 왼쪽으로 1/4, 중심, 중심에서 오른쪽으로 1/4,  왼쪽으로 1/4+1/12, 오른쪽으로 1/4-1/12, 왼쪽으로 1/4-1/12, 오른쪽으로 1/4+1/12  떨어져 있는 점들의 list.
    white_keys = []

    for i in range(52):
        if i == 51:  # C8
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    toppoints[i + 1],
                    toppoints[i],
                    bottompoints[i],
                ]
            )
        elif i == 0:
            j = idx_black.index(i + 1)
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    bottomblackpoints[j][1],
                    bottomblackpoints[j][5],
                    topblackpoints[j][4],
                    toppoints[i],
                    bottompoints[i],
                ]
            )
        elif i % 7 in [2, 5]:  # C,F
            j = idx_black.index(i + 1)
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    bottomblackpoints[j][1],
                    bottomblackpoints[j][3],
                    topblackpoints[j][2],
                    toppoints[i],
                    bottompoints[i],
                ]
            )
        elif i % 7 in [3]:  # D
            j = idx_black.index(i + 1)
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    bottomblackpoints[j][1],
                    bottomblackpoints[j][5],
                    topblackpoints[j][4],
                    topblackpoints[j - 1][3],
                    bottomblackpoints[j - 1][4],
                    bottomblackpoints[j - 1][1],
                    bottompoints[i],
                ]
            )
        elif i % 7 in [0]:  # A
            j = idx_black.index(i + 1)
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    bottomblackpoints[j][1],
                    bottomblackpoints[j][5],
                    topblackpoints[j][4],
                    topblackpoints[j - 1][1],
                    bottomblackpoints[j - 1][2],
                    bottomblackpoints[j - 1][1],
                    bottompoints[i],
                ]
            )
        elif i % 7 in [6]:  # G
            j = idx_black.index(i + 1)
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    bottomblackpoints[j][1],
                    bottomblackpoints[j][0],
                    topblackpoints[j][0],
                    topblackpoints[j - 1][3],
                    bottomblackpoints[j - 1][4],
                    bottomblackpoints[j - 1][1],
                    bottompoints[i],
                ]
            )
        elif i % 7 in [1, 4]:  # E, B
            j = idx_black.index(i)
            white_keys.append(
                [
                    bottompoints[i],
                    bottompoints[i + 1],
                    toppoints[i + 1],
                    topblackpoints[j][5],
                    bottomblackpoints[j][6],
                    bottomblackpoints[j][1],
                    bottompoints[i],
                ]
            )

    black_keys = []
    for i in range(len(idx_black)):
        if idx_black[i] % 7 in [0]:
            black_keys.append(
                [
                    topblackpoints[i][0],
                    bottomblackpoints[i][0],
                    bottomblackpoints[i][2],
                    topblackpoints[i][1],
                    topblackpoints[i][0],
                ]
            )
        elif idx_black[i] % 7 in [1, 4]:
            black_keys.append(
                [
                    topblackpoints[i][4],
                    bottomblackpoints[i][5],
                    bottomblackpoints[i][6],
                    topblackpoints[i][5],
                    topblackpoints[i][4],
                ]
            )
        elif idx_black[i] % 7 in [3, 6]:
            black_keys.append(
                [
                    topblackpoints[i][2],
                    bottomblackpoints[i][3],
                    bottomblackpoints[i][4],
                    topblackpoints[i][3],
                    topblackpoints[i][2],
                ]
            )
    keylist = []
    for i in range(52):
        if i % 7 in [0, 2, 3, 5, 6]:
            keylist.append(white_keys[0])
            white_keys.pop(0)
            if i != 51:
                keylist.append(black_keys[0])
                black_keys.pop(0)
        if i % 7 in [1, 4]:
            keylist.append(white_keys[0])
            white_keys.pop(0)
    return keylist


def handpositiondetector(handsinfo, floatingframes, keylist):
    # floatingframes=detectfloatingframes(handlist,frame_count)
    pressedkeyslist = []
    pressingfingerslist = []
    fingertippositionslist = []
    for hand in handsinfo:
        if [hand.handframe, hand.handtype] in floatingframes:
            pressedkeyslist.append([hand.handtype, "floating"])
            pressingfingerslist.append(["floating"])
            fingertippositionslist.append(["floating"])
            continue
        pressedkeylist = []
        pressedkeylist.append(hand.handtype)
        pressingfingerlist = []
        fingertippositionlist=[]
        for i in [4, 8, 12, 16, 20]:
            for j in range(len(keylist)):
                if (
                    inside_or_outside(
                        keylist[j], [hand.handlandmark[i].x*0.9+hand.handlandmark[i-1].x*0.1, hand.handlandmark[i].y*0.9+hand.handlandmark[i-1].y*0.1]
                    )
                    == 1
                ):
                    pressedkeylist.append(j)
                    pressingfingerlist.append(int(i / 4))
                    fingertippositionlist.append([hand.handlandmark[i].x*0.9+hand.handlandmark[i-1].x*0.1, hand.handlandmark[i].y*0.9+hand.handlandmark[i-1].y*0.1])


        if len(pressedkeylist) <= 1:
            pressedkeylist.append("floating")  # 가끔 안될때가있음
            pressingfingerlist.append("floating")
            fingertippositionlist.append("floating")

        pressedkeyslist.append(pressedkeylist)
        pressingfingerslist.append(pressingfingerlist)
        fingertippositionslist.append(fingertippositionlist)
    if len(handsinfo) == 0:
        pressedkeyslist= ['Noinfo']
        pressingfingerslist= ['Noinfo']
        fingertippositionslist= ['Noinfo']
    return [pressedkeyslist, pressingfingerslist, fingertippositionslist] #[그 frame에서 눌린 key들의 모임(중복 허용: 대부분 열 손가락의 정보가 다있음), 손가락 번호, 손가락 위치]


def inside_or_outside(polygon, point):
    # https://losskatsu.github.io/machine-learning/py-polygon01
    """
    한 점(point)이 다각형(polygon)내부에 위치하는지 외부에 위치하는지 판별하는 함수
    입력값
        polygon -> 다각형을 구성하는 리스트 정보
        point -> 판별하고 싶은 점
    출력값
        내부에 위치하면 res = 1
        외부에 위치하면 res = 0
    """
    N = len(polygon) - 1  # N각형을 의미
    counter = 0
    p1 = polygon[0]
    for i in range(1, N + 1):
        p2 = polygon[i % N]
        if (
            point[1] > min(p1[1], p2[1])
            and point[1] <= max(p1[1], p2[1])
            and point[0] <= max(p1[0], p2[0])
            and p1[1] != p2[1]
        ):
            xinters = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            if p1[0] == p2[0] or point[0] <= xinters:
                counter += 1
        p1 = p2
    if counter % 2 == 0:
        res = 0  # point is outside
    else:
        res = 1  # point is inside
    return res


# STEP 5: Process the classification result. In this case, visualize it.
# floatingframes=detectfloatingframes(handlist,frame_count)
# print(f"faulty frames:{len(nohandframelist)}, floating hands:{floatingframes}")

# # STEP 3: Load the input image.
# image = mp.Image.create_from_file(filepath+"frame_annotated99_NoDetect_squarecrop2.jpg")

# # STEP 4: Detect hand landmarks from the input image.
# detection_result = detector.detect(image)
# print(len(detection_result.hand_landmarks[0]))
# # STEP 5: Process the classification result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2.imshow("test",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
