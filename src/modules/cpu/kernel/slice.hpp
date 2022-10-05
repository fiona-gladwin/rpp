#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void applyPolicy(RpptOutOfBoundsPolicy policyType, Rpp32s *anchor, Rpp32s *sliceEnd, Rpp32s *srcBufferLength)
{
    switch (policyType)
    {
        case RpptOutOfBoundsPolicy::PAD:
            break;
        case RpptOutOfBoundsPolicy::TRIMTOSHAPE:
            *anchor = std::min(std::max(*anchor, 0), *srcBufferLength);
            *sliceEnd = std::min(std::max(*anchor + *sliceEnd, 0), *srcBufferLength);
            break;
        case RpptOutOfBoundsPolicy::ERROR:
        default:
            bool isOutOfBounds = (*anchor < 0) || (*anchor > *srcBufferLength);
            break;
    }
}

RppStatus slice_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32s *srcLengthTensor,
                            Rpp32s *anchorTensor,
                            Rpp32s *dstBufferLengthTensor,
                            Rpp32s *axes,
                            Rpp32f fillValues,
                            bool normalizedAnchor,
                            bool normalizedShape,
                            RpptOutOfBoundsPolicy policyType)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f fillValue = fillValues;
        Rpp32s srcBufferLength = srcLengthTensor[batchCount];
        Rpp32s anchor = anchorTensor[batchCount];
        Rpp32s sliceEnd = anchor + dstBufferLengthTensor[batchCount];
        Rpp32s dstBufferLength = sliceEnd - anchor;

        // If normalized between 0 - 1 convert to actual indices
        if(normalizedAnchor)
            anchor *= dstBufferLength; // To be checked
        if(normalizedShape)
            dstBufferLength *= dstBufferLength; // TO Be checked

        applyPolicy(policyType, &anchor, &sliceEnd, &srcBufferLength);  // check the policy and update the values accordingly
        dstBufferLength = sliceEnd - anchor;

        if(srcDescPtr->c == 1 && anchor == 0 && dstBufferLength == srcBufferLength)
        {
            // Do a memcpy if output dimension input dimension
            memcpy(dstPtrTemp, srcPtrTemp, dstBufferLength * sizeof(Rpp32f));
        }
        else
        {
            Rpp32s vectorIncrement = 8;
            Rpp32s alignedLength = (dstBufferLength / 8) * 8;
            __m256 pFillValue = _mm256_set1_ps(fillValue);

            bool needPad = (anchor < 0) || ((anchor + dstBufferLength) > srcBufferLength);
            Rpp32s dstIdx = 0;
            if (needPad)
            {
                // out of bounds (left side)
                Rpp32s numIndices = std::abs(std::min(anchor, 0));
                Rpp32s leftPadLength = std::min(numIndices, dstBufferLength);
                Rpp32s alignedLeftPadLength = (leftPadLength / 8) * 8;

                for (; dstIdx < alignedLeftPadLength; dstIdx += vectorIncrement)
                {
                    _mm256_storeu_ps(dstPtrTemp, pFillValue);
                    dstPtrTemp += vectorIncrement;
                }
                for (; dstIdx < leftPadLength; dstIdx++)
                {
                    *dstPtrTemp = fillValue;
                    dstPtrTemp++;
                }

                anchor += leftPadLength;
            }

            // within input bounds
            Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
            Rpp32s dstLengthInBounds = std::max(dstBufferLength - dstIdx, 0);
            Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
            memcpy(dstPtrTemp, &srcPtrTemp[anchor], (size_t)(lengthInBounds * sizeof(Rpp32f)));
            dstIdx += lengthInBounds;
            dstPtrTemp += lengthInBounds;

            if (needPad)
            {
                // out of bounds (right side)
                for (; dstIdx < alignedLength; dstIdx += vectorIncrement)
                {
                    _mm256_storeu_ps(dstPtrTemp, pFillValue);
                    dstPtrTemp += vectorIncrement;
                }
                for (; dstIdx < dstBufferLength; dstIdx++)
                {
                    *dstPtrTemp = fillValue;
                    dstPtrTemp++;
                }
            }
        }
    }

	return RPP_SUCCESS;
}

RppStatus slice_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32s *srcLengthTensor,
                            Rpp32s *anchorTensor,
                            Rpp32s *dstBufferLengthTensor,
                            Rpp32s *axes,
                            Rpp32f fillValue,
                            Rpp32s numOfDims,
                            bool normalizedAnchor,
                            bool normalizedShape,
                            RpptOutOfBoundsPolicy policyType)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f fillValue = fillValue;
        Rpp32s sampleBatchCount = batchCount * numOfDims;

        // // If normalized between 0 - 1 convert to actual indices
        // if(normalizedAnchor) {
        //     anchorLength *= dstBufferLength;
        //     anchorChannels *= dstChannels;
        // }
        // if(normalizedShape) { // Doubt
        //     dstBufferLength *= dstBufferLength;
        //     dstChannels *= dstChannels;
        // }
        if(srcDescPtr->c == 1)
        {
            Rpp32s srcBufferLength = srcLengthTensor[batchCount];
            Rpp32s anchor = anchorTensor[batchCount];
            Rpp32s dstBufferLength = dstBufferLengthTensor[batchCount];
            Rpp32s sliceEnd = anchor + dstBufferLength;
            
            applyPolicy(policyType, &anchor, &sliceEnd, &srcBufferLength);  // check the policy and update the values accordingly
            dstBufferLength = sliceEnd - anchor;
            
            if(anchor == 0 && dstBufferLength == srcBufferLength) {
            // Do a memcpy if output dimension input dimension
                memcpy(dstPtrTemp, srcPtrTemp, dstBufferLength * sizeof(Rpp32f));
            } else {
                Rpp32s vectorIncrement = 8;
                Rpp32s alignedLength = (dstBufferLength / 8) * 8;
                __m256 pFillValue = _mm256_set1_ps(*fillValue);

                bool needPad = (anchor < 0) || ((anchor + dstBufferLength) > srcBufferLength);
                Rpp32s dstIdx = 0;
                if (needPad)
                {
                    // out of bounds (left side)
                    Rpp32s numIndices = std::abs(std::min(anchor, 0));
                    Rpp32s leftPadLength = std::min(numIndices, dstBufferLength);
                    Rpp32s alignedLeftPadLength = (leftPadLength / 8) * 8;

                    for (; dstIdx < alignedLeftPadLength; dstIdx += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrTemp, pFillValue);
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; dstIdx < leftPadLength; dstIdx++)
                    {
                        *dstPtrTemp = *fillValue;
                        dstPtrTemp++;
                    }

                    anchor += leftPadLength;
                }

                // within input bounds
                Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
                Rpp32s dstLengthInBounds = std::max(dstBufferLength - dstIdx, 0);
                Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
                memcpy(dstPtrTemp, &srcPtrTemp[anchor], (size_t)(lengthInBounds * sizeof(Rpp32f)));
                dstIdx += lengthInBounds;
                dstPtrTemp += lengthInBounds;

                if (needPad)
                {
                    // out of bounds (right side)
                    for (; dstIdx < alignedLength; dstIdx += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrTemp, pFillValue);
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; dstIdx < dstBufferLength; dstIdx++)
                    {
                        *dstPtrTemp = *fillValue;
                        dstPtrTemp++;
                    }
                }
            }
        } else if(srcDescPtr->c > 1) {
            Rpp32s srcBufferLength = srcLengthTensor[sampleBatchCount + 1];
            Rpp32s anchor = anchorTensor[sampleBatchCount + 1];
            Rpp32s dstBufferLength = dstBufferLengthTensor[sampleBatchCount + 1];
            Rpp32s sliceEnd = anchor + dstBufferLength;
            bool channelsProcessed = false;
            // For the channel dimensiom
            // Check if channel dim for that particular dim is above zero and if it needs to be sliced or not
            if(srcBufferLength > 1 && srcBufferLength != dstBufferLength) {
                applyPolicy(policyType, &anchor, &sliceEnd, &srcBufferLength);  // check the policy and update the values accordingly
                dstBufferLength = sliceEnd - anchor;
                
                bool needPad = (anchor < 0) || ((anchor + dstBufferLength) > srcBufferLength);
                Rpp32s dstIdx = 0;
                channelsProcessed = true;
                for(int row = 0; row < srcLengthTensor[sampleBatchCount]; row++)
                {
                    Rpp32f * dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.wStride;
                    Rpp32f * srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.wStride;
                    if (needPad) {
                        // out of bounds (left side)
                        Rpp32s numIndices = std::abs(std::min(anchor, 0));
                        Rpp32s leftPadLength = std::min(numIndices, dstBufferLength);
                        for (; dstIdx < leftPadLength; dstIdx++) {
                            *dstPtrTempRow = fillValue;
                            dstPtrTempRow++;
                        }
                        anchor += leftPadLength;
                    }
                    
                    Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
                    Rpp32s dstLengthInBounds = std::max(dstBufferLength - dstIdx, 0);
                    Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
                    memcpy(dstPtrTempRow, srcPtrTempRow + anchor, (size_t)(lengthInBounds * sizeof(Rpp32f)));
                    dstIdx += lengthInBounds;
                    dstPtrTempRow += lengthInBounds;
                    
                    if (needPad) {
                        // out of bounds (right side)
                        for (; dstIdx < dstBufferLength; dstIdx++) {
                            *dstPtrTempRow = fillValue;
                            dstPtrTempRow++;
                        }
                    }
                }
            }
            
            srcBufferLength = srcLengthTensor[sampleBatchCount];
            anchor = anchorTensor[sampleBatchCount];
            dstBufferLength = dstBufferLengthTensor[sampleBatchCount];
            sliceEnd = anchor + dstBufferLength;
            int row = 0;
            Rpp32f * srcPtrTempRow = srcPtrTemp;
            if(anchor > 0)
                srcPtrTempRow += anchor * srcDescPtr->strides.wStride;
            Rpp32f * dstPtrTempRow = dstPtrTemp;

            applyPolicy(policyType, &anchor, &sliceEnd, &srcBufferLength);  // check the policy and update the values accordingly
            dstBufferLength = sliceEnd - anchor;
            bool needPad = (anchor < 0) || ((anchor + dstBufferLength) > srcBufferLength);
            Rpp32s dstIdx = 0;
            if (needPad) {
                // out of bounds (left side)
                Rpp32s numIndices = std::abs(std::min(anchor, 0));
                Rpp32s leftPadLength = std::min(numIndices, dstBufferLength);
                for (; dstIdx < leftPadLength; dstIdx++) {
                    memset(dstPtrTempRow, fillValue, dstDescPtr->strides.wStride * sizeof(Rpp32f));
                    dstPtrTempRow += dstDescPtr->strides.wStride;
                }
                anchor += leftPadLength;
            }
            Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
            Rpp32s dstLengthInBounds = std::max(dstBufferLength, 0);
            Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
            for(; dstIdx < lengthInBounds; dstIdx++) {
                memcpy(dstPtrTempRow, srcPtrTempRow, srcDescPtr->strides.wStride * sizeof(Rpp32f));
                srcPtrTempRow += srcDescPtr->strides.wStride;
                dstPtrTempRow += dstDescPtr->strides.wStride;
            }
            if(needPad) {
                // out of bounds (right side)
                for (; dstIdx < dstBufferLength; dstIdx++) {
                    memset(dstPtrTempRow, fillValue, dstDescPtr->strides.wStride * sizeof(Rpp32f));
                    dstPtrTempRow += dstDescPtr->strides.wStride;
                }
            }
        }
    }

	return RPP_SUCCESS;
}
