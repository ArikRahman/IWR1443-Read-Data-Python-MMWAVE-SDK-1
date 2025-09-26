import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

configFileName = "1443config.cfg"
CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15, dtype="uint8")
byteBufferLength = 0


def serialConfig(configFileName):
    global CLIport
    global Dataport
    # MacOS device assignments
    CLIport = serial.Serial("/dev/tty.usbmodemR10310411", 115200)
    Dataport = serial.Serial("/dev/tty.usbmodemR10310414", 921600)
    # may be this just as easily:
    # CLIport = serial.Serial("/dev/tty.usbmodemR10310414", 115200)
    # Dataport = serial.Serial("/dev/tty.usbmodemR10310411", 921600)

    config = [line.rstrip("\r\n") for line in open(configFileName)]
    for i in config:
        CLIport.write((i + "\n").encode())
        print(i)
        time.sleep(0.01)
    return CLIport, Dataport


def parseConfigFile(configFileName):
    configParameters = {}
    config = [line.rstrip("\r\n") for line in open(configFileName)]
    for i in config:
        splitWords = i.split(" ")
        numRxAnt = 4
        numTxAnt = 3
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 *= 2
            digOutSampleRate = int(splitWords[11])
        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(splitWords[5])
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
        2 * freqSlopeConst * 1e12 * numAdcSamples
    )
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
        2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"]
    )
    configParameters["dopplerResolutionMps"] = 3e8 / (
        2
        * startFreq
        * 1e9
        * (idleTime + rampEndTime)
        * 1e-6
        * configParameters["numDopplerBins"]
        * numTxAnt
    )
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (
        2 * freqSlopeConst * 1e3
    )
    configParameters["maxVelocity"] = 3e8 / (
        4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt
    )
    return configParameters


def readAndParseData14xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    maxBufferSize = 2**15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    magicOK = 0
    dataOK = 0
    frameNumber = 0
    detObj = {}
    try:
        readBuffer = Dataport.read(Dataport.in_waiting)
    except Exception:
        return 0, 0, {}
    byteVec = np.frombuffer(readBuffer, dtype="uint8")
    byteCount = len(byteVec)
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength : byteBufferLength + byteCount] = byteVec[
            :byteCount
        ]
        byteBufferLength += byteCount
    if byteBufferLength > 16:
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc : loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)
        if startIdx:
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[: byteBufferLength - startIdx[0]] = byteBuffer[
                    startIdx[0] : byteBufferLength
                ]
                byteBuffer[byteBufferLength - startIdx[0] :] = np.zeros(
                    len(byteBuffer[byteBufferLength - startIdx[0] :]), dtype="uint8"
                )
                byteBufferLength -= startIdx[0]
            if byteBufferLength < 0:
                byteBufferLength = 0
            word = [1, 2**8, 2**16, 2**24]
            totalPacketLen = np.matmul(byteBuffer[12 : 12 + 4], word)
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    if magicOK:
        word = [1, 2**8, 2**16, 2**24]
        idX = 0
        magicNumber = byteBuffer[idX : idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX : idX + 4], word), "x")
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX : idX + 4], word), "x")
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        for tlvIdx in range(numTLVs):
            word = [1, 2**8, 2**16, 2**24]
            tlv_type = np.matmul(byteBuffer[idX : idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX : idX + 4], word)
            idX += 4
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                word = [1, 2**8]
                tlv_numObj = np.matmul(byteBuffer[idX : idX + 2], word)
                idX += 2
                tlv_xyzQFormat = 2 ** np.matmul(byteBuffer[idX : idX + 2], word)
                idX += 2
                rangeIdx = np.zeros(tlv_numObj, dtype="int16")
                dopplerIdx = np.zeros(tlv_numObj, dtype="int16")
                peakVal = np.zeros(tlv_numObj, dtype="int16")
                x = np.zeros(tlv_numObj, dtype="int16")
                y = np.zeros(tlv_numObj, dtype="int16")
                z = np.zeros(tlv_numObj, dtype="int16")
                for objectNum in range(tlv_numObj):
                    rangeIdx[objectNum] = np.matmul(byteBuffer[idX : idX + 2], word)
                    idX += 2
                    dopplerIdx[objectNum] = np.matmul(byteBuffer[idX : idX + 2], word)
                    idX += 2
                    peakVal[objectNum] = np.matmul(byteBuffer[idX : idX + 2], word)
                    idX += 2
                    x[objectNum] = np.matmul(byteBuffer[idX : idX + 2], word)
                    idX += 2
                    y[objectNum] = np.matmul(byteBuffer[idX : idX + 2], word)
                    idX += 2
                    z[objectNum] = np.matmul(byteBuffer[idX : idX + 2], word)
                    idX += 2
                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                dopplerIdx[
                    dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)
                ] = (
                    dopplerIdx[
                        dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)
                    ]
                    - 65535
                )
                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat
                detObj = {
                    "numObj": tlv_numObj,
                    "rangeIdx": rangeIdx,
                    "range": rangeVal,
                    "dopplerIdx": dopplerIdx,
                    "doppler": dopplerVal,
                    "peakVal": peakVal,
                    "x": x,
                    "y": y,
                    "z": z,
                }
                dataOK = 1
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen
            byteBuffer[: byteBufferLength - shiftSize] = byteBuffer[
                shiftSize:byteBufferLength
            ]
            byteBuffer[byteBufferLength - shiftSize :] = np.zeros(
                len(byteBuffer[byteBufferLength - shiftSize :]), dtype="uint8"
            )
            byteBufferLength -= shiftSize
            if byteBufferLength < 0:
                byteBufferLength = 0
    return dataOK, frameNumber, detObj


# ---- MAIN ----

CLIport, Dataport = serialConfig(configFileName)
configParameters = parseConfigFile(configFileName)

app = QtWidgets.QApplication([])
pg.setConfigOption("background", "w")
win = pg.GraphicsLayoutWidget(show=True, title="2D scatter plot")
p = win.addPlot()
p.setXRange(-0.5, 0.5)
p.setYRange(0, 1.5)
p.setLabel("left", text="Y position (m)")
p.setLabel("bottom", text="X position (m)")
s = p.plot([], [], pen=None, symbol="o")
detObj = {}
frameData = {}
currentIndex = 0


def update():
    global detObj, frameData, currentIndex
    dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)
    if dataOk and len(detObj.get("x", [])) > 0:
        x = -detObj["x"]
        y = detObj["y"]
        s.setData(x, y)
        frameData[currentIndex] = detObj
        currentIndex += 1
    print("dataOk:", dataOk)
    print("detObj:", detObj)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(33)  # ~30 Hz

try:
    app.exec_()
except KeyboardInterrupt:
    CLIport.write(("sensorStop\n").encode())
    CLIport.close()
    Dataport.close()
    win.close()
