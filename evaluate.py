from floatinghands import *
from midicomparison import *

midiname = "LisztConsolation3"


def tokenmodifier(tokenlist):
    tempposition = ""
    for token in tokenlist:
        if not "Bar" in token[0]:
            if not "Position" in token[0]:
                token.insert(0, tempposition)
            else:
                tempposition = token[-5]
        else:
            tempposition = token[-5]
    return tokenlist


def loaddata(midiname, frame_rate):
    testtokenlist = miditotoken(midiname + "_test", frame_rate, "highres")  # test
    gttokenlist = miditotoken(midiname + "_gt", frame_rate, "original")  # groundtruth
    testtokenlist = tokenmodifier(testtokenlist)
    gttokenlist = tokenmodifier(gttokenlist)
    return testtokenlist, gttokenlist


def changepositionbypitch(tokenlist):  # Assume there is no tokens of same pitch
    pitchlist = []
    for token in tokenlist:
        pitchlist.append(int(token[-3][6:]))
    pitchlist = sorted(pitchlist)  # pitchlist 정렬

    newtokenlist = []
    for i in range(len(tokenlist)):
        thetoken = [None]
        for token in tokenlist:
            if pitchlist[i] == int(token[-3][6:]):  # pitch 맞추기
                thetoken = token
        newtokenlist.append(
            tokenlist[i][:-4] + thetoken[-4:-2] + tokenlist[i][-2:]
        )  # token 정보 바꾸기
    return newtokenlist


def realign(tokenlist, midires):  # framethreshold: 동시에 친 것을 몇 프레임까지 봐줄까?
    realignedtokenlist = []
    temp = []
    framethreshold = midires // 25
    for token in tokenlist:
        for temptoken in temp:
            if "Bar" in token[0]:
                if (
                    abs(int(temptoken[-5][9:]) - int(token[-5][9:]) + midires)
                    >= framethreshold
                ):  # 이 미디 녹음 환경에선 한마디가 380 position으로 구성됨.
                    temp = changepositionbypitch(temp)  # temp 정렬 후
                    realignedtokenlist += temp  # 최종 리스트에 정렬된 temp 넣기
                    temp = []
            elif abs(int(temptoken[-5][9:]) - int(token[-5][9:])) >= framethreshold:
                temp = changepositionbypitch(temp)  # temp 정렬 후
                realignedtokenlist += temp  # 최종 리스트에 정렬된 temp 넣기
                temp = []
        temp.append(token)
    realignedtokenlist += changepositionbypitch(temp)
    return realignedtokenlist


def pitchsort(tokenlist):
    pitchlist = []
    for token in tokenlist:
        pitchlist.append(int(token[-3][6:]))
    tokenlist = [token for _, token in sorted(zip(pitchlist, tokenlist))]
    return tokenlist


def alignbypitch(tokenlist):  # MIDI 상 화음을 pitch순으로 정렬.
    alignedtokenlist = []
    temp = []
    for token in tokenlist:
        for temptoken in temp:
            print(token)
            if abs(int(temptoken[-5][9:]) - int(token[-5][9:])) >= 1:
                temp = pitchsort(temp)  # temp 정렬 후
                alignedtokenlist += temp  # 최종 리스트에 정렬된 temp 넣기
                temp = []
        temp.append(token)
    alignedtokenlist += pitchsort(temp)
    return alignedtokenlist


def evaluate(testtokenlist, testmidires, gttokenlist):  # midi resolution
    testtokenlist = realign(testtokenlist, testmidires)
    gttokenlist = alignbypitch(gttokenlist)
    l = len(gttokenlist)
    x = 0
    for i in range(len(gttokenlist)):
        if gttokenlist[i][-4:-2] == testtokenlist[i][-4:-2]:
            x += 1
        else:
            print(gttokenlist[i], testtokenlist[i])

    return 100 * x / l


midiname = "miditest_cropped"
testtokenlist, gttokenlist = loaddata(midiname, 30)
print(f"{midiname}: {round(evaluate(testtokenlist,380,gttokenlist),2)}%")
