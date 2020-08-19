import flowio
from topimage import topImageFlow
from midimage import midImageFlow
from botimage import botImageFlow


def getFlowSet(img1, img2, h, w):
    # h and w are the height and width of flow images
    # not original equi image
    flowSet = list()
    flowSet.append(topImageFlow(img1, img2, h, w))
    flowSet.append(midImageFlow(img1, img2, h, w))
    flowSet.append(botImageFlow(img1, img2, h, w))
    return flowSet


def getFlowFileSet(dirName):
    flowSet = []
    topFlowSet = []
    midFlowSet = []
    botFlowSet = []
    # Get NP flows
    for i in range(5):
        tempFlow = flowio.readFlowFile(f"{dirName}/top{i}.flo")
        topFlowSet.append(tempFlow)
    for i in range(10):
        tempFlow = flowio.readFlowFile(f"{dirName}/mid{i}.flo")
        midFlowSet.append(tempFlow)
    for i in range(5):
        tempFlow = flowio.readFlowFile(f"{dirName}/bot{i}.flo")
        botFlowSet.append(tempFlow)
    flowSet.append(topFlowSet)
    flowSet.append(midFlowSet)
    flowSet.append(botFlowSet)

    return flowSet
