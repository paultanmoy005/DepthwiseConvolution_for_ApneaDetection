from scipy.io import loadmat
import numpy as np
import os.path
import pyedflib
import csv

"""
This function divides the signal into segments with duration of 'timeLength' and assign 0 or 1 annotation after each
segment based on the time points in the 'matPath' of the onsets of apneic events.
"""


def dataAnnotation(channel, fs, Header, matPath, timeLength=11, minThresh=1, nonOverlap=False, csvFile=None,
                   save=False):
    """
    :param channel: int, 5 for ECG and 6 for SpO2
    :param fs: int, 128 for ECG and 8 for SpO2
    :param Header: list, list of the record names in the dataset ['ucddb002',....]
    :param matPath: str, path of the .mat file containing the time points of the onset of activity in seconds
    :param timeLength: int, length of each processing window
    :param minThresh: int, the minimum duration of apneic activity for a segment to be considered an apneic segment
    :param nonOverlap: boolean, True if there is no overLap between consecutive processing windows
    :param csvFile: str, path of the output file
    :param save: boolean, True if the output file needs to be saved
    """
    count = 0
    window = fs * timeLength
    timing = loadmat(matPath)
    for header in Header:
        file = header + ".rec"
        if os.path.isfile(file):
            Time = timing['timing_val'][count][0]
            lastApnea = Time[-1][-1]
            f = pyedflib.EdfReader(file)
            sig = list(f.readSignal(channel))[:(fs * lastApnea)]
            if lastApnea * fs <= len(sig):
                storeLabel = np.zeros(lastApnea, 'int')
                for time in Time:
                    storeLabel[time[0] - 1:time[1]] = np.ones(time[1] - time[0] + 1, 'int')
                storeLabel = [1 if sum(storeLabel[i:i + timeLength]) >= minThresh else 0 for i in
                              range(len(storeLabel))][:-(timeLength - 1)]
                data = np.array([sig[i * fs:(i * fs) + window] for i in range(len(storeLabel))])
                if nonOverlap:
                    data = []
                    storeLabel = storeLabel[::timeLength]
                    for i in range(0, len(storeLabel)):
                        data.append(sig[i * fs:(i * fs) + window])

                storeLabel = np.expand_dims(storeLabel, 1)
                data = np.concatenate((data, storeLabel), axis=1)
                if count == 0:
                    finalData = data
                else:
                    finalData = np.concatenate((finalData, data), axis=0)
                print(f'{header} file processed with {len(data)} data')
            count = count + 1
    if save:
        with open(csvFile, mode='w') as csvOutput:
            processed = csv.writer(csvOutput)
            for i in finalData:
                processed.writerow(i)
