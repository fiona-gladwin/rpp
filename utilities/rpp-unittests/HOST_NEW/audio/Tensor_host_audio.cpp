#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>
#include <experimental/filesystem>
#include <iomanip>
// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6

using namespace std;
using half_float::half;

typedef half Rpp16f;

void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

void verify_output(Rpp32f *dstPtr, int *srcLength, int bs, string test_case, Rpp32u stride, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }
        int offset = i * stride;
        int matched_indices = 0;
        for(int j = 0; j < srcLength[i]; j++)
        {
            Rpp32f ref_val, out_val;
            ref_file>>ref_val;
            out_val = dstPtr[offset + j];
            if(abs(out_val - ref_val) < 1e-4)
                matched_indices += 1;
        }
        ref_file.close();
        if(matched_indices == srcLength[i])
            file_match++;
    }

    std::cerr<<std::endl<<"Results for Test case: "<<test_case<<std::endl;
    if(file_match == bs)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<bs<<" outputs are matching with reference outputs"<<std::endl;
}

void verify_non_silent_region_detection(int *detectionIndex, int *detectionLength, string test_case, int bs, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }

        Rpp32s ref_index, ref_length;
        Rpp32s out_index, out_length;
        ref_file>>ref_index;
        ref_file>>ref_length;
        out_index = detectionIndex[i];
        out_length = detectionLength[i];

        if((out_index == ref_index) && (out_length == ref_length))
            file_match += 1;
        ref_file.close();
    }
    std::cerr<<std::endl<<"Results for Test case: "<<test_case<<std::endl;
    if(file_match == bs)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<bs<<" outputs are matching with reference outputs"<<std::endl;
}

void read_spectrogram(Rpp32f *srcPtr, RpptImagePatch *srcDims, int bs, string test_case, Rpp32u stride, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HOST_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }
        int offset = i * stride;
        for(int j = 0; j < srcDims->width * srcDims->height; j++)
        {
            Rpp32f ref_val, out_val;
            ref_file>>ref_val;
            srcPtr[offset + j] = ref_val;
        }
        ref_file.close();
    }
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 3;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_audio <src folder> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:3>\n");
        return -1;
    }

    char *src = argv[1];
    int ip_bitDepth = atoi(argv[2]);
    int test_case = atoi(argv[3]);

    // Set case names
    char funcName[1000];
    switch (test_case)
    {
        case 0:
            strcpy(funcName, "non_silent_region_detection");
            break;
        case 1:
            strcpy(funcName, "to_decibels");
            break;
        case 2:
            strcpy(funcName, "pre_emphasis_filter");
            break;
        case 3:
            strcpy(funcName, "down_mixing");
            break;
        case 4:
            strcpy(funcName, "slice");
            break;
        case 5:
            strcpy(funcName, "mel_filter_bank");
            break;
        case 6:
            strcpy(funcName, "normalize");
            break;
        default:
            strcpy(funcName, "test_case");
            break;
    }

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst data types in tensor descriptors
    if (ip_bitDepth == 2)
    {
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxChannels = 0, maxLength = 0;
    int maxDstLength = 0;
    unsigned long long count = 0;
    unsigned long long iBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);
    printf("\nRunning %s...", func);

    // Get number of audio files
    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfAudioFiles += 1;
    }
    closedir(dr);

    // Initialize the AudioPatch for source
    Rpp32s *inputAudioSize = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));

    // Set maxLength
    char audioNames[noOfAudioFiles][1000];

    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        {
            sf_close (infile);
            continue;
        }

        inputAudioSize[count] = sfinfo.frames * sfinfo.channels;
        srcLengthTensor[count] = sfinfo.frames;
        channelsTensor[count] = sfinfo.channels;
        maxLength = std::max(maxLength, srcLengthTensor[count]);
        maxChannels = std::max(maxChannels, channelsTensor[count]);

        // Close input
        sf_close (infile);
        count++;
    }
    closedir(dr);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfAudioFiles;
    dstDescPtr->n = noOfAudioFiles;

    srcDescPtr->h = 1;
    dstDescPtr->h = 1;

    srcDescPtr->w = maxLength;
    dstDescPtr->w = maxLength;

    srcDescPtr->c = maxChannels;
    if(test_case == 3)
        dstDescPtr->c = 1;
    else
        dstDescPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    iBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    i = 0;
    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        Rpp32f *input_temp_f32;
        input_temp_f32 = inputf32 + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        {
            sf_close (infile);
            continue;
        }

        int bufferLength = sfinfo.frames * sfinfo.channels;
        if(ip_bitDepth == 2)
        {
            readcount = (int) sf_read_float (infile, input_temp_f32, bufferLength);
            if(readcount != bufferLength)
                std::cerr<<"F32 Unable to read audio file completely"<<std::endl;
        }
        i++;

        // Close input
        sf_close (infile);
    }
    closedir(dr);

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, srcDescPtr->n);
    clock_t start, end;
    double start_omp, end_omp;
    double cpu_time_used, omp_time_used;

    string test_case_name;
    switch (test_case)
    {
        case 0:
        {
            test_case_name = "non_silent_region_detection";
            Rpp32s detectionIndex[noOfAudioFiles];
            Rpp32s detectionLength[noOfAudioFiles];
            Rpp32f cutOffDB[noOfAudioFiles];
            Rpp32s windowLength[noOfAudioFiles];
            Rpp32f referencePower[noOfAudioFiles];
            Rpp32s resetInterval[noOfAudioFiles];
            bool referenceMax[noOfAudioFiles];

            for (i = 0; i < noOfAudioFiles; i++)
            {
                cutOffDB[i] = -60.0;
                windowLength[i] = 3;
                referencePower[i] = 1.0;
                resetInterval[i] = -1;
                referenceMax[i] = true;
            }

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_non_silent_region_detection_host(inputf32, srcDescPtr, inputAudioSize, detectionIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, referenceMax, handle);
            }
            else
                missingFuncFlag = 1;

            verify_non_silent_region_detection(detectionIndex, detectionLength, test_case_name, noOfAudioFiles, audioNames);
            break;
        }
        case 1:
        {
            test_case_name = "to_decibels";
            Rpp32f cutOffDB = -200.0;
            Rpp32f multiplier = 10.0;
            Rpp32f referenceMagnitude = 0.0;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_to_decibels_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, cutOffDB, multiplier, referenceMagnitude);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, srcLengthTensor, noOfAudioFiles, test_case_name, dstDescPtr->strides.nStride, audioNames);
            break;
        }
        case 2:
        {
            test_case_name = "pre_emphasis_filter";
            Rpp32f coeff[noOfAudioFiles];
            for (i = 0; i < noOfAudioFiles; i++)
                coeff[i] = 0.97;
            RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_pre_emphasis_filter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, inputAudioSize, coeff, borderType);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, srcLengthTensor, noOfAudioFiles, test_case_name, dstDescPtr->strides.nStride, audioNames);
            break;
        }
        case 3:
        {
            test_case_name = "down_mixing";
            bool normalizeWeights = false;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, normalizeWeights);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, srcLengthTensor, noOfAudioFiles, test_case_name, dstDescPtr->strides.nStride, audioNames);
            break;
        }
        case 4:
        {
            test_case_name = "slice";
            bool normalizedAnchor = false;
            bool normalizedShape = false;
            Rpp32s anchor[noOfAudioFiles];
            Rpp32s shape[noOfAudioFiles];
            Rpp32f fillValues[noOfAudioFiles];
            Rpp32s axes = 0;
            Rpp32s numOfDims = srcDescPtr->c == 1 ? 1 : 2;
            RpptOutOfBoundsPolicy policyType = RpptOutOfBoundsPolicy::PAD;
            
            Rpp32s *srcDimsTensor = (Rpp32s *) calloc(noOfAudioFiles * numOfDims, sizeof(Rpp32s));
            
            for (i = 0; i < noOfAudioFiles * numOfDims; i+=numOfDims)
            {
                srcDimsTensor[i] = srcLengthTensor[i];
                srcDimsTensor[i + 1] = channelsTensor[i];
                shape[i] = 100;
                shape[i + 1] = 2;
                for(int d = 0; d < numOfDims; d++)
                {
                    anchor[i + d] = 0;
                    fillValues[i] = 0.5f;   
                }
            }

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_slice_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDimsTensor, anchor, shape, &axes, fillValues, numOfDims, normalizedAnchor, normalizedShape, policyType);
            }
            else
                missingFuncFlag = 1;

            std::cerr<<"printing output values"<<std::endl;
            Rpp32s size = numOfDims == 1 ? shape[0] : shape[0] * shape[1];
            for(int i = 0; i < size; i++)
                std::cerr<<std::setprecision(11)<<outputf32[i]<<endl;


            verify_output(outputf32, shape, noOfAudioFiles, test_case_name, dstDescPtr->strides.nStride, audioNames);
            break;
        }
        case 5:
        {
            test_case_name = "mel_filter_bank";

            RpptImagePatch *srcDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));
            srcDims[0].width = 225;
            srcDims[0].height = 257;
            Rpp32f sampleRate = 16000;
            Rpp32f minFreq = 0.0;
            Rpp32f maxFreq = sampleRate / 2;
            RpptMelScaleFormula melFormula = RpptMelScaleFormula::SLANEY;
            Rpp32s numFilter = 128;
            bool normalize = true;

            Rpp32f *test_inputf32 = (Rpp32f *)calloc(srcDims[0].width * srcDims[0].height, sizeof(Rpp32f));
            Rpp32f *test_outputf32 = (Rpp32f *)calloc(numFilter * srcDims[0].width, sizeof(Rpp32f));
            read_spectrogram(test_inputf32, srcDims, noOfAudioFiles, "spectrogram", 0, audioNames);

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_mel_filter_bank_host(test_inputf32, srcDescPtr, test_outputf32, dstDescPtr, srcDims, maxFreq, minFreq, melFormula, numFilter, sampleRate, normalize);
            }
            else
                missingFuncFlag = 1;

            int shape[1] = {numFilter * srcDims[0].width};
            verify_output(test_outputf32, shape, noOfAudioFiles, test_case_name, dstDescPtr->strides.nStride, audioNames);

            // std::cerr<<"printing output values"<<std::endl;
            // for(int i = 0; i < numFilter * srcDims[0].width ; i++)
            //     std::cerr<<std::setprecision(11)<<test_outputf32[i]<<endl;

            free(srcDims);
            free(test_inputf32);
            free(test_outputf32);
            break;
        }
        case 6:
        {
            test_case_name = "normalize";
            Rpp32s axis_mask = 2;
            Rpp32f mean, std_dev, scale, shift, epsilon;
            mean = std_dev = scale = shift = epsilon = 0.0f;
            Rpp32s ddof = 0;
            Rpp32s num_of_dims = 2;
            
            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_normalize_audio_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, axis_mask, 
                                          mean, std_dev, scale, shift, epsilon, ddof, num_of_dims);
            }
            else
                missingFuncFlag = 1;

            std::cerr<<"printing output values"<<std::endl;
            for(int i = 0; i < 10 ; i++)
                std::cerr<<std::setprecision(11) << outputf32[i]<<endl;


            // verify_output(outputf32, shape, noOfAudioFiles, test_case_name, dstDescPtr->strides.nStride, audioNames);
            break;
        }
        default:
        {
            missingFuncFlag = 1;
            break;
        }
    }

    end = clock();
    end_omp = omp_get_wtime();

    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    // Display measured times
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    omp_time_used = end_omp - start_omp;
    cout << "\nCPU Time - Tensor: " << cpu_time_used;
    cout << "\nOMP Time - Tensor: " << omp_time_used;
    printf("\n");

    rppDestroyHost(handle);

    // Free memory
    free(inputAudioSize);
    free(srcLengthTensor);
    free(channelsTensor);
    free(inputf32);
    free(outputf32);

    return 0;
}
