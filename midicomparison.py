from miditok import REMI, TokenizerConfig
from symusic import Score
from floatinghands import *
import math
# STEP 1: Import the necessary modules.
from shapely.geometry import Polygon, Point
import geopandas
import sys

sys.path.append("..")


# Creating a multitrack tokenizer configuration, read the doc to explore other parameters


# Loads a midi, converts to tokens, and back to a MIDI


def generatetokenizer(file_name, fps, highres=False):
    filepath = "./midiconvert/"
    print(f"midi: {filepath}{file_name}.mid")
    midi = Score(filepath + file_name + ".mid")
    if not midi.tempos:
        tempo = 120
    else:
        tempo = midi.tempos[0].qpm
    beatres = math.floor(fps * 60 / tempo)
    if highres == True:
        beatres = 96
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=False,
        use_programs=True,
        beat_res={(0, 100): beatres},
    )
    tokenizer = REMI(config)
    tokens = tokenizer(midi)
    return tokenizer, beatres, tokens


# token frame별로 자르기
def miditotoken(file_name, fps, mode):  # mode: original or simplified
    # Loads a midi, converts to tokens, and back to a MIDI
    # calling the tokenizer will automatically detect MIDIs, paths and tokens
    if mode == "highres":
        tokenizer, beatres, tokens = generatetokenizer(file_name, fps, highres=True)
        tokenlist = []
        tempindex = 0
        for i in range(len(tokens.tokens)):
            if "Duration" in tokens.tokens[i]:
                tokenlist.append(tokens.tokens[tempindex : i + 1])
                tempindex = i + 1
        return tokenlist  # [(Bar), Position, Program, Pitch, Velocity, Duration]
    tokenizer, beatres, tokens = generatetokenizer(file_name, fps)
    tokenlist = []
    tempindex = 0
    for i in range(len(tokens.tokens)):
        if "Duration" in tokens.tokens[i]:
            tokenlist.append(tokens.tokens[tempindex : i + 1])
            tempindex = i + 1

    if mode == "original":
        return tokenlist
    for token in tokenlist:
        while "Bar" in token[0]:
            token.pop(0)
            token.append("Bar")
    positionindex = 0
    for i in range(len(tokenlist)):
        token = tokenlist[i]
        if not "Position" in token[0]:
            token.insert(0, tokenlist[positionindex][0])
        else:
            positionindex = i
    barcounter = -1
    # 여기까지 token list 의 형태: [Position(0~4*Beat_res-1), Program(Type of instrument), Pitch(0~128), Velocity, Duration(m.n.beat_res), Bar(if this exists, then the bar has changed) ]
    for token in tokenlist:
        while "Bar" in token[-1]:
            token.pop(-1)
            barcounter += 1
        token[0] = int(token[0][9:]) + 4 * beatres * barcounter

    for token in tokenlist:
        token.pop(1)
        token[1] = int(token[1][6:]) - 21
        token.pop(2)
        token[2] = (
            token[0]
            + int(token[2].split(".")[0][9:]) * beatres
            + int(token[2].split(".")[1])
        )
    for i in range(len(tokenlist)):
        tokenlist[i].append(i)
    if mode == "simplified":
        return tokenlist


def tokentoframeinfo(tokenlist, frame_count):
    # 여기까지 token list 의 형태: [Total Start Position, Pitch(0~95(가상건반은 G8까지 있음)), End position , token number]
    framemidilist = []
    for frame in range(frame_count):
        keylist = []
        for token in tokenlist:
            if frame in range(token[0], token[2]):
                keylist.append([token[1], token[3]])
        framemidilist.append(keylist)
    return framemidilist
    # Output: frame별 [눌린 키, token number]

def betweendotpolygon(dot, polygon):
    poly=Polygon([(vertices[0], vertices[1]) for vertices in polygon])
    gpoly=geopandas.GeoSeries(poly)
    point=geopandas.GeoSeries([Point(dot[0], dot[1])])
    return float(gpoly.distance(point).iloc[0])
def keydistance(keyboard, key, fingertipposition):
    distance = 0

    dlist=[]

    if inside_or_outside(keyboard[key], fingertipposition)==1:
        return 0
    elif inside_or_outside(keyboard[key], fingertipposition)!=1:
        dlist.append(betweendotpolygon(fingertipposition, keyboard[key]))
    distance += min(dlist)
    return distance

def onsetcoefficient(keyonset, key, frame, mode='off'):
    if mode == 'off':
        return 1
    if frame in keyonset[key]:
        return 2
    else:
        return 0.75
def handfingercorresponder(framemidilist, framehandfingerlist, keyboard, tokenlist):
    # framehandfingerlist[i][0]: framehandlist
    # framehandfingerlist[i][1]: framefingerlist
    # framehandfingerlist[i][2]: fingertippositionlist
    #Extract onset frames
    keyonset={}
    for token in tokenlist:
        if token[1] in keyonset.keys():
            keyonset[token[1]]+=list(range(token[0],int(token[0]+0.2*(token[2]-token[0]))))
        else:
            keyonset[token[1]]=list(range(token[0],int(token[0]+0.2*(token[2]-token[0]))))

    keyhandlist = []
    handtypes = ["Left", "Right"]
    halfkeyboarddistance=(keyboard[0][1][0]-keyboard[0][0][0])/2
    
    frame=0
    for _ in stqdm(range(len(framemidilist)), desc="Correponding frame images to midi..."):
        framekeylist = framemidilist[frame]
        for key in framekeylist:  # 각 frame에서 눌려져 있는 상태의 각 key마다
            mindiff = 88
            mindiffhand = "Noinfo"
            fingercount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            handspositioninfo = framehandfingerlist[frame][0]
            fingertippositionsinfo = framehandfingerlist[frame][2]
            handcounter = 0
            for i in range(len(handspositioninfo)):  # 해당 frame의 각 손마다
                handpositioninfo=handspositioninfo[i]
                fingertippositioninfo=fingertippositionsinfo[i]
                if len(handpositioninfo) <= 1 or framehandfingerlist[frame][1][0] == "Noinfo":
                    continue
                if handpositioninfo[1] != "floating":
                    for j in range(1, len(handpositioninfo)):
                        if abs(key[0] - handpositioninfo[j]) < mindiff:
                            mindiff = abs(key[0] - handpositioninfo[j])
                            mindiffhand = handpositioninfo[0]
                        if handpositioninfo[0] == "Left":
                            if (abs(key[0] - handpositioninfo[j]) == 0):  # 손가락과 frame midi 일치
                                fingercount[
                                    framehandfingerlist[frame][1][
                                        handspositioninfo.index(handpositioninfo)
                                    ][j - 1]
                                ] += onsetcoefficient(keyonset, key[0], i, mode="off")*1  # 1~5
                            
                            elif keydistance(keyboard, key[0], fingertippositioninfo[j-1])<halfkeyboarddistance:  # 손가락과 frame midi 반 건반 오차 (euclidean distance) (0.5만큼 보정)
                                fingercount[
                                    framehandfingerlist[frame][1][
                                        handspositioninfo.index(handpositioninfo)
                                    ][j - 1]
                                ] += onsetcoefficient(keyonset, key[0], i, mode="off")*(1-keydistance(keyboard, key[0], fingertippositioninfo[j-1])/halfkeyboarddistance)**2 # 1~5
                        if handpositioninfo[0] == "Right":
                            if (abs(key[0] - handpositioninfo[j]) < 1):  # 손가락과 frame midi 일치
                                fingercount[
                                    framehandfingerlist[frame][1][
                                        handspositioninfo.index(handpositioninfo)
                                    ][j - 1]
                                    + 5
                                ] += onsetcoefficient(keyonset, key[0], i, mode="off")*1  # 6~10
                            elif keydistance(keyboard, key[0], fingertippositioninfo[j-1])<halfkeyboarddistance:  # 손가락과 frame midi 반 건반 오차 (euclidean distance) (0.5만큼 보정)
                                fingercount[
                                    framehandfingerlist[frame][1][
                                        handspositioninfo.index(handpositioninfo)
                                    ][j - 1]
                                    + 5
                                ] += onsetcoefficient(keyonset, key[0], i, mode="off")*(1-keydistance(keyboard, key[0], fingertippositioninfo[j-1])/halfkeyboarddistance)**2  # 6~10
                    handcounter += 1
                if handpositioninfo[1] == "floating":
                    mindiffhand = handtypes[handtypes.index(handpositioninfo[0]) - 1]
            if handcounter == 0:
                mindiffhand = "Noinfo"
            if (
                handcounter == 1 and mindiff > 1 and mindiffhand != "Noinfo"
            ):  # 오차범위: 반음
                mindiffhand = handtypes[handtypes.index(mindiffhand) - 1]
            key.append(mindiffhand)
            key.append(fingercount)
        keyhandlist.append(framekeylist)
        frame+=1
    return keyhandlist  # [[key, tokennumber, hand, fingercount]]


pitch_list = [
    "A0","A#0","B0",
    "C1","C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1",
    "C2","C#2","D2","D#2","E2","F2","F#2","G2","G#2","A2","A#2","B2",
    "C3","C#3","D3","D#3","E3","F3","F#3","G3","G#3","A3","A#3","B3",
    "C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4",
    "C5","C#5","D5","D#5","E5","F5","F#5","G5","G#5","A5","A#5","B5",
    "C6","C#6","D6","D#6","E6","F6","F#6","G6","G#6","A6","A#6","B6",
    "C7","C#7","D7","D#7","E7","F7","F#7","G7","G#7","A7","A#7","B7",
    "C8",
]

# 각 token마다 가장 많이 채택된 손 대응하기
"""
def handdecider(tokenlist,keyhandlist,fps):
    tokenhandlist=[]
    for i in range(len(tokenlist)):
        #tokenlist[i].pop(0)  #Total Start Position (frame)
        #tokenlist[i].pop(1)  # End position
        #tokenlist[i][0]=pitch_list[tokenlist[i][0]+12]
        tokenlist[i].pop(2)  # End position
        tokenlist[i][1]=pitch_list[tokenlist[i][1]+12]
        lefthandcounter=0
        righthandcounter=0
        noinfocounter=0
        lhindex=[]
        rhindex=[]
        fingerindex=[[],[],[],[],[],[],[],[],[],[],0] # 10개의 손가락과 Noinfo counter
        for j in range(len(keyhandlist)):
            framekeyhandinfo=keyhandlist[j]
            for keyhandinfo in framekeyhandinfo:
                if keyhandinfo[1]==i:
                    if keyhandinfo[2]=="Left":
                        lefthandcounter +=1
                        lhindex.append(j)
                        for k in range(1,11):
                            if keyhandinfo[3][k]!=0: fingerindex[k-1].append(j)
                        if keyhandinfo[3]==[0,0,0,0,0,0,0,0,0,0,0]: fingerindex[10]+=1
                    elif keyhandinfo[2] == "Right":
                        righthandcounter +=1
                        rhindex.append(j)
                        for k in range(1,11):
                            if keyhandinfo[3][k]!=0: 
                                fingerindex[k-1].append(j)
                        if keyhandinfo[3]==[0,0,0,0,0,0,0,0,0,0,0]: fingerindex[10]+=1
                    elif keyhandinfo[2] == "Noinfo":
                        noinfocounter +=1
                        fingerindex[10]+=1
        counterlist=[lefthandcounter,righthandcounter,noinfocounter,(lefthandcounter+righthandcounter+noinfocounter)/2]
        maxfinger="Noinfo"
        maxfingercount=fingerindex[10]
        if counterlist.index(max(counterlist))==0:
            tokenlist[i].append("Left")
            for j in range(5):
                if len(fingerindex[j])>maxfingercount:
                    maxfingercount=len(fingerindex[j])
                    maxfinger=j+1
        elif counterlist.index(max(counterlist))==1:
            tokenlist[i].append("Right")
            for j in range(5,10):
                if len(fingerindex[j])>maxfingercount:
                    maxfingercount=len(fingerindex[j])
                    maxfinger=j+1
        else:
            tokenlist[i].append("Noinfo")
        fingercount=[len(fingerindex[i]) for i in range(0,10)]
        #if lefthandcounter>0 and righthandcounter>0 or noinfocounter>0:
             #print(f"Tokennumber:{i},   Tokenpitch={pitch_list[tokenlist[i][1]]},    lefthandcounter={lhindex},     righthandcounter={rhindex}, noinfocounter ={noinfocounter}")
        print(f"Tokennumber:{i},   Tokenpitch={tokenlist[i][1]},    lefthandcounter={len(lhindex)},     righthandcounter={len(rhindex)}, noinfocounter ={noinfocounter}, fingercount={fingercount} fingernoinfo={fingerindex[10]}")
        pressedfingers=[]
        for j in range(10):
            if fingercount[j] != 0: pressedfingers.append(j)

        if len(pressedfingers) >=2:
            maxfinger=input(f"Choose the actual finger which pressed {tokenlist[i][1]} at frame {tokenlist[i][0]} or time {int(tokenlist[i][0]/fps//60)}:{tokenlist[i][0]/fps%60} : ")
        tokenlist[i].append(maxfinger)
        tokenhandlist.append(tokenlist[i])
        
    return tokenhandlist  #Output: [Pitch(0~95(가상건반은 C0~ G8까지 있음)), token number, hand, finger]
"""


def handdecider(tokenlist, keyhandlist, fps):
    tokenhandlist = []
    for i in range(len(tokenlist)):
        # tokenlist[i].pop(0)  #Total Start Position (frame)
        # tokenlist[i].pop(1)  # End position
        # tokenlist[i][0]=pitch_list[tokenlist[i][0]+12]
        tokenlist[i].pop(2)  # End position
        tokenlist[i][1] = pitch_list[tokenlist[i][1]]
        lefthandcounter = 0
        righthandcounter = 0
        noinfocounter = 0
        lhindex = []
        rhindex = []
        fingerindex = [[],[],[],[],[],[],[],[],[],[],0,
        ]  # 10개의 손가락과 Noinfo counter
        for j in range(len(keyhandlist)):
            framekeyhandinfo = keyhandlist[j]
            for keyhandinfo in framekeyhandinfo:
                if keyhandinfo[1] == i:
                    if keyhandinfo[2] == "Left":
                        lefthandcounter += 1
                        lhindex.append(j)
                        for k in range(1, 11):
                            if keyhandinfo[3][k] != 0:
                                fingerindex[k - 1].append(j)
                        if keyhandinfo[3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                            fingerindex[10] += 1
                    elif keyhandinfo[2] == "Right":
                        righthandcounter += 1
                        rhindex.append(j)
                        for k in range(1, 11):
                            if keyhandinfo[3][k] != 0:
                                fingerindex[k - 1].append(j)
                        if keyhandinfo[3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                            fingerindex[10] += 1
                    elif keyhandinfo[2] == "Noinfo":
                        noinfocounter += 1
                        fingerindex[10] += 1
        counterlist = [
            lefthandcounter,
            righthandcounter,
            noinfocounter,
            (lefthandcounter + righthandcounter + noinfocounter) / 2,
        ]
        if counterlist.index(max(counterlist)) == 0:
            tokenlist[i].append("Left")

        elif counterlist.index(max(counterlist)) == 1:
            tokenlist[i].append("Right")

        else:
            tokenlist[i].append("Noinfo")
        fingercount = [len(fingerindex[i]) for i in range(0, 10)]
        # if lefthandcounter>0 and righthandcounter>0 or noinfocounter>0:
        # print(f"Tokennumber:{i},   Tokenpitch={pitch_list[tokenlist[i][1]]},    lefthandcounter={lhindex},     righthandcounter={rhindex}, noinfocounter ={noinfocounter}")
        print(
            f"Tokennumber:{i},   Tokenpitch={tokenlist[i][1]},    lefthandcounter={len(lhindex)},     righthandcounter={len(rhindex)}, noinfocounter ={noinfocounter}, fingercount={fingercount} fingernoinfo={fingerindex[10]}"
        )
        pressedfingers = []
        pressedfinger = 0
        for j in range(10):
            if fingercount[j] != 0:
                pressedfingers.append(j + 1)
        if len(pressedfingers) == 0:
            pressedfinger = "Noinfo"
        if len(pressedfingers) == 1:
            pressedfinger = pressedfingers[0]
        if len(pressedfingers) >= 2:
            pressedfinger = input(
                f"Choose the actual finger among {pressedfingers} which pressed {tokenlist[i][1]} at frame {tokenlist[i][0]} or time {str(int(tokenlist[i][0]/fps//60)).zfill(2)}:{str(math.floor(tokenlist[i][0]/fps%60)).zfill(2)}:"
                + format(
                    math.floor(tokenlist[i][0] / fps % 60 * 1000) / 1000
                    - math.floor(tokenlist[i][0] / fps % 60),
                    ".2f",
                )[2:]
                + " : "
            )
        tokenlist[i].append(pressedfinger)
        tokenhandlist.append(tokenlist[i])

    return tokenhandlist  # Output: [Pitch(0~95(가상건반은 C0~ G8까지 있음)), token number, hand, finger]
