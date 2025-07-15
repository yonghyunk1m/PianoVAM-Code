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
HANDEDNESS_TEXT_COLOR = (255, 235, 0)  # Middle yellow

def draw_landmarks_and_floatedness_on_image(
    rgb_image, detection_result, frame, floating_handinfos
):

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

def landmarkdistance(landmarka, landmarkb, ratio):
    return ((landmarka.x - landmarkb.x) ** 2 + (landmarka.y*ratio - landmarkb.y*ratio) ** 2) ** 0.5 * 2 # Multiply 2 because of change of range of area coorditates from [0,1] to [-1,1]

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

def faultyframes(handlist):  # Exclude frames w/ more than three hands or two hands with same handtype
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
                    thehand = hand 
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
            ): 
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
    for metric in metriclist:
        if (
            metric[2] < 0.9
        ):  # Differs from how many and how intensive of floating occurs in the whole playing
            floatingframes.append([metric[0], metric[1], metric[2], 'floating'])
        else: floatingframes.append([metric[0], metric[1], metric[2], 'notfloating'])

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
): 

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

    # topblackpoints: List of points from 1/4 key left and right, 1/4+1/12 key left, 1/4-1/12 key right, 1/4-1/12 key left and 1/4 + 1/12 key right from the center(the line between white keys)
    # bottomblackpoints: List of points from 1/4 key left, center, 1/4 key right, 1/4+1/12 key left, 1/4-1/12 key right, 1/4-1/12 key left and 1/4 + 1/12 key right from the center(the line between white keys)

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
            pressedkeylist.append("floating") 
            pressingfingerlist.append("floating")
            fingertippositionlist.append("floating")

        pressedkeyslist.append(pressedkeylist)
        pressingfingerslist.append(pressingfingerlist)
        fingertippositionslist.append(fingertippositionlist)
    if len(handsinfo) == 0:
        pressedkeyslist= ['Noinfo']
        pressingfingerslist= ['Noinfo']
        fingertippositionslist= ['Noinfo']
    return [pressedkeyslist, pressingfingerslist, fingertippositionslist] #[Pressed keys list in that frame of handsinfo, fingering number, finger position]


def inside_or_outside(polygon, point):
    # https://losskatsu.github.io/machine-learning/py-polygon01

    N = len(polygon) - 1  # N-gon
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
