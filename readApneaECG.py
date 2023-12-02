import csv
import wfdb
from scipy.signal import resample

def apneaECGData(path, outFileEcg, outFileSpO2, outFileMatchedEcg, timeLength = 11, save=False):
    """
    :param path: str, path of the directory with apneaECG records
    :param outFileEcg: str, output file name with ECG segments
    :param outFileSpO2: str, output file name with SpO2 segments
    :param outFileMatchedEcg: str, output file name with ECG segments that co-occur with SpO2 segments
    :param save: boolean, True if the files need to be saved
    """
    A1 = ["a0%d" % i for i in range(1, 10)]
    A2 = ["a%d" % i for i in range(10, 21)]
    B = ["b0%d" % i for i in range(1, 6)]
    C = ["c0%d" % i for i in range(1, 10)]
    X1 = ["x0%d" % i for i in range(1, 10)]
    X2 = ["x%d" % i for i in range(10, 36)]
    Record = [A1 + A2 + B + C + ["c10"] + X1 + X2][0]
    storeSegment = []
    storeAnnotation = []
    storeSegmentSpo2 = []
    storeAnnotationSpo2 = []
    storeSegmentMatchECG = []
    minute = 6000
    fs = 100
    segmentLength = int(timeLength * fs)
    resampLengthECG = timeLength * 128
    resampLengthSPO2 = timeLength * 8
    for record in Record:
        fileIn = path + record
        annot_apnea = wfdb.io.rdann(fileIn, 'apn')
        annotation = annot_apnea.symbol
        signalEcg = wfdb.io.rdsamp(fileIn)[0]

        for j in range(len(annotation)):
            minuteSignalEcg = signalEcg[j * minute:j * minute + minute].ravel()
            ecgSeg = list(minuteSignalEcg[:segmentLength])

            if record in ['a01', 'a02', 'a03', 'a04', 'b01', 'c01', 'c02', 'c03']:
                signalSpo2 = wfdb.io.rdsamp(fileIn + 'r')[0][:, 3]
                minuteSignalSpo2 = signalSpo2[j * minute:j * minute + minute]
                spo2Seg = list(minuteSignalSpo2[:segmentLength])
                if len(spo2Seg) == segmentLength and len(spo2Seg) == len(ecgSeg):
                    ecgSeg = list(resample(ecgSeg, resampLengthECG))
                    ecgSeg.append(annotation[j])
                    spo2Seg = list(resample(spo2Seg, resampLengthSPO2))
                    spo2Seg.append(annotation[j])
                    storeSegmentMatchECG.append(ecgSeg)
                    storeAnnotation += annotation[j]
                    storeSegmentSpo2.append(spo2Seg)
                    storeAnnotationSpo2 += annotation[j]

            if len(ecgSeg) == segmentLength:
                ecgSeg = list(resample(ecgSeg, resampLengthECG))
                ecgSeg.append(annotation[j])
                storeSegment.append(ecgSeg)
                storeAnnotation += annotation[j]

    if save:
        with open(outFileEcg, mode='w') as csvOutput:
            processed = csv.writer(csvOutput)
            for i in storeSegment:
                processed.writerow(i)

        with open(outFileSpO2, mode='w') as csvOutput:
            processed = csv.writer(csvOutput)
            for i in storeSegmentSpo2:
                processed.writerow(i)

        with open(outFileMatchedEcg, mode='w') as csvOutput:
            processed = csv.writer(csvOutput)
            for i in storeSegmentMatchECG:
                processed.writerow(i)
