# DepthwiseSeparableConvolution_for_ApneaDetection
Depthwise separable convolution (DSC) is an efficient type of convolution that splits the standard convolution operation into two separate steps:
Depthwise Convolution: Applies a single filter to each input channel independently.
Pointwise Convolution: Applies a 1×1 convolution to combine the output of the depthwise convolution across channels.
This decomposition reduces the computational cost and number of parameters compared to standard convolutions, making it particularly useful in lightweight deep learning architectures such as MobileNet and EfficientNet.

In this project, DSC architecture is used to detect apneic events using St. Vincent Hospital's Dataset and Apnea-ECG Dataset from the PhysioNet Databank. Unlike conventional spatial convolution, DSC requires fewer number of parameters and fewer number of floating point of operations.

The following variables in *main.py* are needed to be set accordingly:

Header = Header of each record from the St. Vincent Hospital Dataset
matPath = .mat file containing the continuous annotation from St. Vincent Hospital dataset
timeLength = length of each signal segment
minThresh = minmum seconds with apneic activity within each segment
path = path to the dataset
fileVincent_ECG = path to the .csv file with ECG signal from St. Vincet Hospital dataset
fileVincent_SpO2 = path to the .csv file with SpO2 signal from St. Vincent Hospital dataset
fileApnea_ECG = path to the .csv file with ECG signal from Apnea-ECG dataset
fileApnea_matchedECG = path to the .csv file with ECG signal from Apnea-ECG dataset that have corresponding SpO2 data points
fileApnea_SpO2 = path to the .csv file with SpO2 signal from Apnea-ECG dataset
