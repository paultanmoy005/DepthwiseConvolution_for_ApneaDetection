# DepthwiseSeparableConvolution_for_ApneaDetection
Depthwise separable convolution (DSC) is an efficient type of convolution that splits the standard convolution operation into two separate steps:
Depthwise Convolution: Applies a single filter to each input channel independently.
Pointwise Convolution: Applies a 1×1 convolution to combine the output of the depthwise convolution across channels.
This decomposition reduces the computational cost and number of parameters compared to standard convolutions, making it particularly useful in lightweight deep learning architectures such as MobileNet and EfficientNet.

In this project, DSC architecture is used to detect apneic events. Unlike conventional spatial convolution, DSC requires fewer number of parameters and fewer number of floating point of operations.

