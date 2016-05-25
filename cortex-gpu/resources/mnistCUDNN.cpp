/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.

 * The sample can work in single, double, half precision, but it
 * assumes the data in files is stored in single precision
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include "ImageIO.h"
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "gemv.h"
#include "error_util.h"

#define IMAGE_H 28
#define IMAGE_W 28

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *ip1_bin = "ip1.bin";
const char *ip1_bias_bin = "ip1.bias.bin";
const char *ip2_bin = "ip2.bin";
const char *ip2_bias_bin = "ip2.bias.bin";

/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

#define EXIT_WAIVED 0


void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};
template <> struct ScaleFactorTypeMap<half1>  { typedef float Type;};

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
    template <class T>
    value_type operator()(T x) {return value_type(x);}
    value_type operator()(half1 x) {return value_type(cpu_half2float(x));}
};

template <>
class Convert<half1>
{
public:
    template <class T>
    half1 operator()(T x) {return cpu_float2half_rn (T(x));} 
    half1 operator()(half1 x) {return x;}
};

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }
    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }
    // conversion
    Convert<value_type> fromReal;
    for (int i = 0; i < size; i++)
    {
        data_h[i] = fromReal(data_tmp[i]);
    }
    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

template <class value_type>
void readImage(const char* fname, value_type* imgData_h)
{
    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    std::string sFilename(fname);
    std::cout << "Loading image " << sFilename << std::endl;
    // Take care of half precision
    Convert<value_type> fromReal;
    // load gray-scale image from disk
    try
    {
        npp::loadImage(sFilename, oHostSrc);
    }
    catch (npp::Exception &rException)
    {
        FatalError(rException.toString());
    }
    // Plot to console and normalize image to be in range [0,1]
    for (int i = 0; i < IMAGE_H; i++)
    {
        for (int j = 0; j < IMAGE_W; j++)
        {   
            int idx = IMAGE_W*i + j;
            imgData_h[idx] = fromReal(*(oHostSrc.data() + idx) / double(255));
        }
    } 
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    Convert<real_type> toReal;
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << toReal(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2
 } fp16Import_t;
template <class value_type>
struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST){};
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        fp16Import = _fp16Import;
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
private:
    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }
};

template <>
void Layer_t<half1>::readAllocInit(const char* fname, int size, half1** data_h, half1** data_d)
{
    *data_h = new half1[size];
    int size_b = size*sizeof(half1);
    checkCudaErrors( cudaMalloc(data_d, size_b) );    
    float *data_tmp_h, *data_tmp_d;

    switch(fp16Import)
    {
        case FP16_HOST :
        {
            readBinaryFile<half1>(fname, size, *data_h);
            checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b,
                                cudaMemcpyHostToDevice) );
            break;
        }
        case FP16_CUDA :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);

            gpu_float2half_rn<float>(size, data_tmp_d, *data_d);

            delete [] data_tmp_h;
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
        case FP16_CUDNN :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
            delete [] data_tmp_h;
            cudnnHandle_t cudnnHandle;
            cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
            checkCUDNN( cudnnCreate(&cudnnHandle) );
            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
                                                CUDNN_DATA_FLOAT,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
                                                CUDNN_DATA_HALF,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            float alpha = 1.0f;
            float beta = 0.0f;
            checkCUDNN( cudnnTransformTensor(cudnnHandle, &alpha,
                                             srcTensorDesc,
                                             data_tmp_d, &beta,
                                             dstTensorDesc,
                                             *data_d) );
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
            checkCUDNN( cudnnDestroy(cudnnHandle) );
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
    }
}

// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}

template <class value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }
    void destroyHandles()
    {
        checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCublasErrors( cublasDestroy(cublasHandle) );
    }
  public:
    network_t()
    {
        convAlgorithm = -1;
        switch (sizeof(value_type))
        {
            case 2 : dataType = CUDNN_DATA_HALF; break;
            case 4 : dataType = CUDNN_DATA_FLOAT; break;
            case 8 : dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }
    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(1);
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }
    void fullyConnectedForward(const Layer_t<value_type>& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        resize(dim_y, dstData);

        scaling_type alpha = scaling_type(1), beta = scaling_type(1);
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta,*dstData);

        h = 1; w = 1; c = dim_y;
    }
    void convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              tensorDims,
                                              filterDimA) );
 
        const int convDims = 2;
        int padA[convDims] = {0,0};
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
        }
        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        if (convAlgorithm < 0)
        {
            // Choose the best according to the preference
            std::cout << "Testing cudnnGetConvolutionForwardAlgorithm ...\n";
            checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &algo
                                                    ) );
            std::cout << "Fastest algorithm is Algo " << algo << "\n";
            convAlgorithm = algo;
            // New way of finding the fastest config
            // Setup for findFastest call
            std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
            int requestedAlgoCount = 5; 
            int returnedAlgoCount[1];
            cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
            checkCUDNN(cudnnFindConvolutionForwardAlgorithm( cudnnHandle, 
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     convDesc,
                                                     dstTensorDesc,
                                                     requestedAlgoCount,
                                                     returnedAlgoCount,
                                                     results
                                                   ) );
        for(int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex){
            printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
        }
            free(results);
        }
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
            if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            {
                //std::cout << "Using FFT for convolution\n";
            }
        }

        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData)
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                poolDims,
                                                windowDimA,
                                                paddingA,
                                                strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                    srcTensorDesc,
                                                    tensorDims,
                                                    tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
     
        resize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            CUDNN_ACTIVATION_RELU,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }

    int classify_example(const char* fname, const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& ip1,
                          const Layer_t<value_type>& ip2)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        readImage(fname, imgData_h);

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        convoluteForward(conv1, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        convoluteForward(conv2, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        fullyConnectedForward(ip1, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);
        lrnForward(n, c, h, w, srcData, &dstData);

        fullyConnectedForward(ip2, n, c, h, w, dstData, &srcData);
        softmaxForward(n, c, h, w, srcData, &dstData);

        const int max_digits = 10;
        // Take care of half precision
        Convert<scaling_type> toReal;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (toReal(result[id]) < toReal(result[i])) id = i;
        }

        std::cout << "Resulting weights from Softmax:" << std::endl;
        printDeviceVector(n*c*h*w, dstData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }
};

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
                          int& n, int& c, int& h, int& w,
                          half1* srcData, half1** dstData)
{
    c = c*h*w; h = 1; w = 1;
    network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData);
    c = ip.outputs;
}
#endif

void displayUsage()
{
    printf( "mnistCUDNN {<options>}\n");
    printf( "help                   : display this help\n");
    printf( "device=<int>           : set the device to run the sample\n");
    printf( "image=<name>           : classify specific image\n");
}

int main(int argc, char *argv[])
{   
    std::string image_path;
    int i1,i2,i3;

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayUsage();
        exit(EXIT_WAIVED); 
    }

    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
    showDevices();

    int device = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        checkCudaErrors( cudaSetDevice(device) );
    }
    std::cout << "Using device " << device << std::endl;
    
    if (checkCmdLineFlag(argc, (const char **)argv, "image"))
    {
        char* image_name;
        getCmdLineArgumentString(argc, (const char **)argv,
                                 "image", (char **) &image_name);        

        network_t<float> mnist;
        Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
        Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
        Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
        Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
        int i1 = mnist.classify_example(image_name, conv1, conv2, ip1, ip2);
        std::cout << "\nResult of classification: " << i1 << std::endl;

        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    // default behaviour
    if (argc == 1 || (argc == 2) && checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        // check available memory
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
        double globalMem = prop.totalGlobalMem/double(1024*1024);
        bool low_memory = false;
        if (globalMem < 1536) 
        {
        // takes care of 1x1 convolution workaround for fully connected layers
        // when CUDNN_CONVOLUTION_FWD_ALGO_FFT is used
#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
            low_memory = true;
#endif
        }
        {
            std::cout << "\nTesting single precision\n";
            network_t<float> mnist;
            Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
            Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
            Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
            Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
            get_path(image_path, first_image, argv[0]);
            i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, second_image, argv[0]);
            i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, third_image, argv[0]);
            // New feature in cuDNN v3: FFT for convolution
            mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
            i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

            std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
            if (i1 != 1 || i2 != 3 || i3 != 5)
            {
                std::cout << "\nTest failed!\n";
                FatalError("Prediction mismatch");
            }
            else
            {
                std::cout << "\nTest passed!\n";
            }
        }

        {
            std::cout << "\nTesting half precision (math in single precision)\n";
            network_t<half1> mnist;
            // Conversion of input weights to half precision is done
            // on host using tools from fp16_emu.cpp
            Layer_t<half1> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0],FP16_HOST);
            Layer_t<half1> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0],FP16_HOST);
            // Conversion of input weights to half precision is done
            // on device using cudnnTransformTensor
            Layer_t<half1>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0], FP16_CUDNN);
            // Conversion of input weights to half precision is done
            // on device using CUDA kernel from fp16_dev.cu
            Layer_t<half1>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0], FP16_CUDA);
            get_path(image_path, first_image, argv[0]);
            i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, second_image, argv[0]);
            i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, third_image, argv[0]);
            // New feature in cuDNN v3: FFT for convolution
            if (!low_memory)
            {
                mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
            }
            i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

            std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
            if (i1 != 1 || i2 != 3 || i3 != 5)
            {
                std::cout << "\nTest failed!\n";
                FatalError("Prediction mismatch");
            }
            else
            {
                std::cout << "\nTest passed!\n";
            }
        }

        cudaDeviceReset();
        exit(EXIT_SUCCESS);        
    }

    displayUsage();
    cudaDeviceReset();
    exit(EXIT_WAIVED);
}
