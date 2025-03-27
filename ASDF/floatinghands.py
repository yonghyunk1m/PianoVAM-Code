from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import numpy as np
from stqdm import stqdm
import math
from scipy.optimize import fsolve
from shapely import Polygon

# STEP 1: Import the necessary modules.
import sys
sys.set_int_max_str_digits(15000)   # sympy 계산할때 이거 limit에 걸려서 늘려줌

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (140, 171, 138)  # pastel green


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
    # 감성적 스타일 커스텀 설
    CUSTOM_LANDMARK_STYLE = solutions.drawing_utils.DrawingSpec(color=(255, 182, 193), thickness=6, circle_radius=5)  # 파스텔핑
    CUSTOM_CONNECTION_STYLE = solutions.drawing_utils.DrawingSpec(color=(135, 206, 235), thickness=3)  # 스카이블루
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    depth=1
    floatedness=''
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        for hand in floating_handinfos:
            if hand[:2]==[frame, handedness[0].category_name]:
                depth=hand[2]
                floatedness=hand[3]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=(landmark.x+1)/2, y=(landmark.y+1)/2, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=CUSTOM_LANDMARK_STYLE,
            connection_drawing_spec=CUSTOM_CONNECTION_STYLE,
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [(landmark.x+1)/2 for landmark in hand_landmarks]
        y_coordinates = [(landmark.y+1)/2 for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        floattext = ""
        if floatedness=="floating": floattext="Float,"
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name},{floattext}{round(depth,3)}",
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
        self.handdepth = 1  #default
    
    def set_handdepth(self, handdepth):
        self.handdepth = handdepth

    #class handdepthclass:
        #    def __init__(self, handtype, depthsum, handframe):
        #        self.handtype = handtype
        #        self.depthsum = depthsum
#        self.handframe = handframe


def landmarkdistance(landmarka, landmarkb, ratio):
    return ((landmarka.x - landmarkb.x) ** 2 + (landmarka.y*ratio - landmarkb.y*ratio) ** 2) ** 0.5 * 2 # [0,1]이 아니라 [-1,1]의 거리이므로 x2 해줘야됨

def landmarkangle(landmarka, landmarkb, landmarkc):
    # Vector AB = B - A
    AB_x = landmarkb.x - landmarka.x
    AB_y = landmarkb.y - landmarka.y

    # Vector BC = C - B
    BC_x = landmarkc.x - landmarkb.x
    BC_y = landmarkc.y - landmarkb.y

    # Dot product of AB and BC
    dot_product = AB_x * BC_x + AB_y * BC_y

    # Magnitudes of AB and BC
    magnitude_AB = math.sqrt(AB_x**2 + AB_y**2)
    magnitude_BC = math.sqrt(BC_x**2 + BC_y**2)

    # Cosine of the angle
    cos_theta = dot_product / (magnitude_AB * magnitude_BC)

    # Angle in radians
    theta_radians = math.acos(cos_theta)

    # Convert radians to degrees
    theta_degrees = math.degrees(theta_radians)

    area = 0.5 * abs(landmarka.x*(landmarkb.y - landmarkc.y) + landmarkb.x*(landmarkc.y - landmarka.y) + landmarkc.x*(landmarka.y - landmarkb.y))

    return theta_degrees, area

def modelskeleton(handlist):
    lhangledifflist = [] # Angle of Index finger MCP - Wrist - Ring finger MCP
    rhangledifflist = []

    for handsinfo in handlist:
        for hand in handsinfo:
            if hand.handtype == "Left":
                lhangledifflist.append(
                    [
                            hand.handtype, 
                            abs(landmarkangle(hand.handlandmark[5],hand.handlandmark[0],hand.handlandmark[13])[0]-28), # difference of angle between 28 degree (the angle when hand is parallel to the keyboard)
                            landmarkangle(hand.handlandmark[5],hand.handlandmark[0],hand.handlandmark[13])[1], # WIR triangle area
                            Polygon(((hand.handlandmark[0].x,hand.handlandmark[0].y),
                                    (hand.handlandmark[4].x,hand.handlandmark[4].y),
                                    (hand.handlandmark[8].x,hand.handlandmark[8].y),
                                    (hand.handlandmark[12].x,hand.handlandmark[12].y),
                                    (hand.handlandmark[16].x,hand.handlandmark[16].y),
                                    (hand.handlandmark[20].x,hand.handlandmark[20].y),      
                            )).area, # area of hexagon of wrist and five fingertips
                            hand.handlandmark,
                            hand.handframe
                    ]
                )
            elif hand.handtype == "Right":
                 rhangledifflist.append(
                    [
                            hand.handtype, 
                            abs(landmarkangle(hand.handlandmark[5],hand.handlandmark[0],hand.handlandmark[13])[0]-28), 
                            landmarkangle(hand.handlandmark[5],hand.handlandmark[0],hand.handlandmark[13])[1],
                            Polygon(((hand.handlandmark[0].x,hand.handlandmark[0].y),
                                    (hand.handlandmark[4].x,hand.handlandmark[4].y),
                                    (hand.handlandmark[8].x,hand.handlandmark[8].y),
                                    (hand.handlandmark[12].x,hand.handlandmark[12].y),
                                    (hand.handlandmark[16].x,hand.handlandmark[16].y),
                                    (hand.handlandmark[20].x,hand.handlandmark[20].y),      
                            )).area, # area of hexagon of wrist and five fingertips
                            hand.handlandmark,
                            hand.handframe
                    ]
                )               
    lhatop10=sorted(lhangledifflist, key=lambda x : x[1])[:round(0.1*len(lhangledifflist))]  # top 10 percent of hands whose angle is close to 28 degree (to exclude tilted hand)
    lhwtop50=sorted(lhatop10, key=lambda x : -x[3]/x[2])[:round(0.5*len(lhatop10))] # top 50 percent of hands by WIR area / whole area (to exclude bended hand)
    lhmodel=sorted(lhwtop50, key=lambda x : x[2])[int(len(lhwtop50)*0.5)][4] #  median of area (to select hand which is likely on keyboard)
    print(f"lhmodel={sorted(lhwtop50, key=lambda x : x[2])[int(len(lhwtop50)*0.5)][5]}")

    rhatop10=sorted(rhangledifflist, key=lambda x : x[1])[:round(0.1*len(rhangledifflist))]
    rhwtop50=sorted(rhatop10, key=lambda x : -x[3]/x[2])[:round(0.5*len(rhatop10))]
    rhmodel=sorted(rhwtop50, key=lambda x : x[2])[int(len(rhwtop50)*0.5)][4]
    print(f"rhmodel={sorted(rhwtop50, key=lambda x : x[2])[int(len(rhwtop50)*0.5)][5]}") 
    return lhmodel, rhmodel
    
def calcdepth(w,i,r,lhmodel,rhmodel,ratio): #ratio: ratio of the video frame = height/width

    initial_guess=[1,1,1]
    solution = fsolve(system, initial_guess, args=(w,i,r,lhmodel,rhmodel,ratio))
    
    t, u, v = solution
    return (t+u+v)/3


def system(vars,w,i,r,lhmodel, rhmodel, ratio):
    t, u, v = vars
    eq1=(t*w[0]-u*i[0])**2+(t*w[1]-u*i[1])**2+(t-u)**2 - landmarkdistance(lhmodel[0],lhmodel[5],ratio)**2
    eq2=(u*i[0]-v*r[0])**2+(u*i[1]-v*r[1])**2+(u-v)**2 - landmarkdistance(lhmodel[5],lhmodel[13],ratio)**2
    eq3=(v*r[0]-t*w[0])**2+(v*r[1]-t*w[1])**2+(v-t)**2 - landmarkdistance(lhmodel[13],lhmodel[0],ratio)**2

    return [eq1, eq2, eq3]



    #def avgcalc(handlist, ratio):
    #    handdistancelist = []
    #
    #    for handsinfo in handlist:
    #        handdistanceinfo = []
    #        for hand in handsinfo:
    #            distancesum = 0
    #            for i in [1, 5, 9, 13, 17]:  # OR, for i in range(21):
    #                distancesum += landmarkdistance(
    #                    hand.handlandmark[0], hand.handlandmark[i], ratio
    #                )
    #                # distancesum += 0.5*landmarkdistance(hand.handlandmark[5],hand.handlandmark[17])
    #            handdistanceinfo.append(
    #                distanceclass(
    #                    handtype=hand.handtype,
    #                    distancesum=distancesum,
    #                    handframe=hand.handframe,
    #                )
    #            )
    #        handdistancelist.append(handdistanceinfo)
    #    avg = 0
    #    avgcounter = 0
    #    for i in handdistancelist:
    #        for hand in i:
    #            avg += hand.distancesum
    #            avgcounter += 1
    #
    #    avg = avg / avgcounter
    #
    #    """lowavg = 0
    #    lowavgcounter = 0
    #    for i in handdistancelist:
    #        for hand in i:
    #            if hand.distancesum <= avg:
    #                lowavg += hand.distancesum
    #                lowavgcounter += 1
    #    lowavg = lowavg / lowavgcounter"""
#    return handdistancelist


def faultyframes(handlist):  # 손 3개 or 같은손 2개 등 
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

def depthlist(handlist,lhmodel,rhmodel,ratio):
    for hands in handlist:
        for hand in hands:
            depth=calcdepth(
                    [hand.handlandmark[0].x, hand.handlandmark[0].y],
                    [hand.handlandmark[5].x, hand.handlandmark[5].y],
                    [hand.handlandmark[13].x, hand.handlandmark[13].y],
                    lhmodel,
                    rhmodel,
                    ratio
            )
            hand.set_handdepth(depth)

def mymetric(
    handlist, handtype, frame, frame_count, a, c, faultyframes, lhmodel, rhmodel, ratio
):
    framerange = [*range(int(max(0, frame - a)), int(min(frame + a, frame_count)))]
    l = len(framerange)
    tempframes = []
    availableframes = []
    templist = []
    thehand = None
    value = 0
    counter=0
    for hands in handlist:
        for hand in hands:
            if hand.handframe in framerange:
                if hand.handtype == handtype:
                    tempframes.append(hand.handframe)
            if hand.handframe == frame:
                if hand.handtype == handtype:
                    thehand = hand  ## 코드 속도를 위해 수정해야할듯, 이부분은 따로 뺄 필요가 있음
    for frames in tempframes:
        if frames not in faultyframes:
            availableframes.append(frames)
    
    thehanddepth=calcdepth(
        [thehand.handlandmark[0].x, thehand.handlandmark[0].y],
        [thehand.handlandmark[5].x, thehand.handlandmark[5].y],
        [thehand.handlandmark[13].x, thehand.handlandmark[13].y],
        lhmodel,
        rhmodel,
        ratio
    )
    for hands in handlist:
        for hand in hands:
            templist = [
                *range(int(max(0, frame - a)), int(min(frame + a, frame_count)))
            ]
            if (
                hand.handframe in framerange
            ):  ## 여기의 hand들은 전부 framerange 안에 있음.
                depth=hand.handdepth
                if hand.handtype == handtype:
                    for availableframe in availableframes:
                        if availableframe < frame:
                            value += depth * (a - frame + availableframe)
                            counter += a - frame + availableframe
                            templist.remove(availableframe)
                        else:
                            value += depth * (a + frame - availableframe)
                            counter += a + frame - availableframe
                            templist.remove(availableframe)
                    for leftframe in templist:
                        if leftframe < frame:
                            value += thehanddepth * (a - frame + leftframe)
                            counter += a - frame + leftframe
                        else:
                            value += thehanddepth * (a + frame - leftframe)
                            counter += a + frame - leftframe
    
    value /= counter

    return ((c * thehanddepth) + (1 - c) * value)

def detectfloatingframes(handlist, frame_count, faultyframes, lhmodel, rhmodel, ratio):
    metriclist = []
    floatingframes = []
    # handdistancelist = avgcalc(handlist, ratio)
    index=0
    for _ in stqdm(range(len(handlist)), desc="Detecting floating hands..."):
        handsinfo = handlist[index]
        for hand in handsinfo:
            metriclist.append(
                [
                    hand.handframe,
                    hand.handtype,
                    mymetric(
                        handlist,
                        hand.handtype,
                        hand.handframe,
                        frame_count,
                        7,
                        0.5,
                        faultyframes,
                        lhmodel,
                        rhmodel,
                        ratio
                    ),
                ]
            )
        index+=1
    #    metricmedian=sorted(metriclist, key=lambda x : x[-1])[int(len(metriclist)/2)][2]
    #    
    #    bighandcounter = 0
    #    for metric in metriclist:
    #        if metric[2] < metricmedian * 0.7:   #bighand: 영상에 포함된 앞뒤의 매우 큰 손 부분을 잘라주기 위함.
    #            bighandcounter += 1
    for metric in metriclist:
        if (
            metric[2] < 0.9
        ):  # Differs from how many and how intensive of floating occurs in the whole playing
            floatingframes.append([metric[0], metric[1], metric[2], 'floating'])
        else: floatingframes.append([metric[0], metric[1], metric[2], 'notfloating'])

    #영상 길이가 짧으면 결과가 이상하게 나와서
    #floatingframes=[]

    return floatingframes


def distortioncoefficient(i, ldistortion, rdistortion, cdistortion):
    value=0
    if i > 26:
        value = rdistortion * (i - 26) / abs(i - 26) * (-((abs(i - 26) - 13) ** 2) + 169) + cdistortion * (26**2-(i-26)**2) / 26**2 
    elif i < 26:
        value = ldistortion * (i - 26) / abs(i - 26) * (-((abs(i - 26) - 13) ** 2) + 169) + cdistortion * (26**2-(i-26)**2) / 26**2
    elif i == 26:
        value = cdistortion
    return value

    ## Fits in our (MACLAB@KAIST) recording environment.


def generatekeyboard(
    lu, ru, ld, rd, blackratio, ldistortion=0, rdistortion=0, cdistortion=0,
):  # 05/12 update: 검은 건반을 가운데가 아니라 실제 건반에 맞게 위치 조정
    # 05/14 update: 카메라 각도에 맞게 검은 건반 위치 미세 조정
    # 07/02 update: x축 distortion 보정

    idx_black = [n for n in range(1, 52) if ((n) % 7) in [0, 1, 3, 4, 6]]

    bottompoints = [
        [
            ld[0] * (52 - i) / 52
            + rd[0] * (i) / 52
            + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
            ld[1] * (52 - i) / 52 + rd[1] * i / 52+0.01,
        ]
        for i in range(53)
    ]

    toppoints = [
        [
            lu[0] * (52 - i) / 52
            + ru[0] * (i) / 52
            + distortioncoefficient(52 - i, ldistortion, rdistortion, cdistortion),
            lu[1] * (52 - i) / 52 + ru[1] * i / 52-0.01,
        ]
        for i in range(53)
    ]

    topblackpoints = [
        [
            [
                lu[0] * (52 - i + 1 / 4 ) / 52
                + ru[0] * (i - 1 / 4 ) / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                lu[1] * (52 - i + 1 / 4) / 52 + ru[1] * (i - 1 / 4) / 52-0.01,
            ],
            [
                lu[0] * (52 - i - 1 / 4) / 52
                + ru[0] * (i + 1 / 4) / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                lu[1] * (52 - i - 1 / 4) / 52 + ru[1] * (i + 1 / 4) / 52-0.01,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 + 1 / 12)
                / 52
                + ru[0]
                * (i - 1 / 4 - 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                lu[1] * (52 - i + 1 / 4 + 1 / 12) / 52
                + ru[1] * (i - 1 / 4 - 1 / 12) / 52-0.01,
            ],
            [
                lu[0]
                * (52 - i - 1 / 4 + 1 / 12)
                / 52
                + ru[0]
                * (i + 1 / 4 - 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                lu[1] * (52 - i - 1 / 4 + 1 / 12) / 52
                + ru[1] * (i + 1 / 4 - 1 / 12) / 52-0.01,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 - 1 / 12)
                / 52
                + ru[0]
                * (i - 1 / 4 + 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                lu[1] * (52 - i + 1 / 4 - 1 / 12) / 52
                + ru[1] * (i - 1 / 4 + 1 / 12) / 52-0.01,
            ],
            [
                lu[0]
                * (52 - i - 1 / 4 - 1 / 12)
                / 52
                + ru[0]
                * (i + 1 / 4 + 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                lu[1] * (52 - i - 1 / 4 - 1 / 12) / 52
                + ru[1] * (i + 1 / 4 + 1 / 12) / 52-0.01,
            ],
        ]
        for i in idx_black
    ]

    bottomblackpoints = [
        [
            [
                lu[0] * (52 - i + 1 / 4) / 52
                + ru[0] * (i - 1 / 4) / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                (lu[1] * (52 - i + 1 / 4) / 52 + ru[1] * (i - 1 / 4) / 52)
                * (1 - blackratio)
                + (ld[1] * (52 - i + 1 / 4) / 52 + rd[1] * (i - 1 / 4) / 52)
                * blackratio,
            ],
            [
                lu[0] * (52 - i) / 52
                + ru[0] * (i) / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                (lu[1] * (52 - i) / 52 + ru[1] * (i) / 52) * (1 - blackratio)
                + (ld[1] * (52 - i) / 52 + rd[1] * (i) / 52) * blackratio,
            ],
            [
                lu[0] * (52 - i - 1 / 4) / 52
                + ru[0] * (i + 1 / 4) / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
                (lu[1] * (52 - i - 1 / 4) / 52 + ru[1] * (i + 1 / 4) / 52)
                * (1 - blackratio)
                + (ld[1] * (52 - i - 1 / 4) / 52 + rd[1] * (i + 1 / 4) / 52)
                * blackratio,
            ],
            [
                lu[0]
                * (52 - i + 1 / 4 + 1 / 12)
                / 52
                + ru[0]
                * (i - 1 / 4 - 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
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
                * (52 - i - 1 / 4 + 1 / 12)
                / 52
                + ru[0]
                * (i + 1 / 4 - 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
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
                * (52 - i + 1 / 4 - 1 / 12)
                / 52
                + ru[0]
                * (i - 1 / 4 + 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
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
                * (52 - i - 1 / 4 - 1 / 12)
                / 52
                + ru[0]
                * (i + 1 / 4 + 1 / 12)
                / 52
                + distortioncoefficient(i, ldistortion, rdistortion, cdistortion),
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
    floatinghands = [info[:2].append(info[3]) for info in floatingframes]
    pressedkeylist=[]
    for hand in handsinfo:
        if [hand.handframe, hand.handtype, 'floating'] in floatinghands:
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
                        keylist[j], [(hand.handlandmark[i].x+1)*0.5*0.9+(hand.handlandmark[i-1].x+1)*0.5*0.1, (hand.handlandmark[i].y+1)*0.5*0.9+(hand.handlandmark[i-1].y+1)*0.5*0.1] # [-1.1] (hand landmark) -> [0,1] (keyboard)
                    )
                    == 1
                ):
                    pressedkeylist.append(j)
                    pressingfingerlist.append(int(i / 4))
                    fingertippositionlist.append([(hand.handlandmark[i].x+1)*0.5*0.9+(hand.handlandmark[i-1].x+1)*0.5*0.1, (hand.handlandmark[i].y+1)*0.5*0.9+(hand.handlandmark[i-1].y+1)*0.5*0.1])
        
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