#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void glitch_pkd_tensor(T *srcPtr,
                                  uint2 srcStridesNH,
                                  T *dstPtr,
                                  uint2 dstStridesNH,
                                  unsigned int *x_offset_r,
                                  unsigned int *y_offset_r,
                                  unsigned int *x_offset_g,
                                  unsigned int *y_offset_g,
                                  unsigned int *x_offset_b,
                                  unsigned int *y_offset_b,
                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int x_r, y_r, x_g, y_g, x_b, y_b;

    x_r = id_x + x_offset_r[id_z];
    y_r = id_y + y_offset_r[id_z];

    x_g = id_x + x_offset_g[id_z];
    y_g = id_y + y_offset_g[id_z];

    x_b = id_x + x_offset_b[id_z];
    y_b = id_y + y_offset_b[id_z];

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    dstPtr[dstIdx] = srcPtr[srcIdx];
    dstPtr[dstIdx + 1] = srcPtr[srcIdx + 1];
    dstPtr[dstIdx + 2] = srcPtr[srcIdx + 2];

    if((y_r >= 0) && (y_r < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdxR = (id_z * srcStridesNH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        dstPtr[dstIdx] = srcPtr[srcIdxR];
    }

    if((y_g >= 0) && (y_g <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdxG = (id_z * srcStridesNH.x) + ((y_g + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (x_g + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3 + 1;
        dstPtr[dstIdx + 1] = srcPtr[srcIdxG];
    }

    if((y_b >= 0) && (y_b <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdxB = (id_z * srcStridesNH.x) + ((y_b + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (x_b + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3  + 2;
        dstPtr[dstIdx + 2] = srcPtr[srcIdxB];
    }
}

template <typename T>
__global__ void glitch_pln_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      unsigned int *x_offset_r,
                                      unsigned int *y_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *y_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_b,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    
    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }
    uint src_pix_idx, dst_pix_idx;
    src_pix_idx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    dst_pix_idx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstPtr[dst_pix_idx] = srcPtr[src_pix_idx];
    dstPtr[dst_pix_idx + dstStridesNCH.y] = srcPtr[src_pix_idx + srcStridesNCH.y];
    dstPtr[dst_pix_idx + dstStridesNCH.y + dstStridesNCH.y] = srcPtr[src_pix_idx + srcStridesNCH.y + srcStridesNCH.y];

    int x_r, x_g, x_b, y_r, y_g, y_b;

    // R
    x_r = (id_x + x_offset_r[id_z]);
    y_r = (id_y + y_offset_r[id_z]);

    // G
    x_g = (id_x + x_offset_g[id_z]);
    y_g = (id_y + y_offset_g[id_z]);

    // B
    x_b = (id_x + x_offset_b[id_z]);
    y_b = (id_y + y_offset_b[id_z]);

    // R
    if ((y_r >= 0) && (y_r <= roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        dstPtr[dst_pix_idx] = srcPtr[ (id_z * srcStridesNCH.x) + (x_r + y_r * srcStridesNCH.z) + indextmp * srcStridesNCH.y];
        indextmp += 1;
        dst_pix_idx += dstStridesNCH.y;
    }

    // G
    if ((y_g >= 0) && (y_g <= roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        dstPtr[dst_pix_idx] = srcPtr[(id_z * srcStridesNCH.x) + (x_g + y_g * srcStridesNCH.z) + indextmp * srcStridesNCH.y];
        indextmp = indextmp + 1;
        dst_pix_idx += dstStridesNCH.y;
    }

    // B
    if ((y_b >= 0) && (y_b <= roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        unsigned char B = srcPtr[(id_z * srcStridesNCH.x) + (x_b + y_b * srcStridesNCH.z) + indextmp * srcStridesNCH.y];
        dstPtr[dst_pix_idx] = B;
    }
}

template <typename T>
__global__ void glitch_pkd3_pln3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      unsigned int *x_offset_r,
                                      unsigned int *y_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *y_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_b,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int x_r, y_r, x_g, y_g, x_b, y_b;

    x_r = id_x + x_offset_r[id_z];
    y_r = id_y + y_offset_r[id_z];

    x_g = id_x + x_offset_g[id_z];
    y_g = id_y + y_offset_g[id_z];

    x_b = id_x + x_offset_b[id_z];
    y_b = id_y + y_offset_b[id_z];

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    dstPtr[dstIdx] = srcPtr[srcIdx];
    dstPtr[dstIdx + dstStridesNCH.y] = srcPtr[srcIdx + 1];
    dstPtr[dstIdx + 2 * dstStridesNCH.y] = srcPtr[srcIdx + 2];

    if((y_r >= 0) && (y_r < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdxR = (id_z * srcStridesNH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        dstPtr[dstIdx] = srcPtr[srcIdxR];
    }

    if((y_g >= 0) && (y_g <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdxG = (id_z * srcStridesNH.x) + ((y_g + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (x_g + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3 + 1;
        dstPtr[dstIdx + dstStridesNCH.y] = srcPtr[srcIdxG];
    }

    if((y_b >= 0) && (y_b <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdxB = (id_z * srcStridesNH.x) + ((y_b + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (x_b + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3  + 2;
        dstPtr[dstIdx + 2 * dstStridesNCH.y] = srcPtr[srcIdxB];
    }
}

template <typename T>
__global__ void glitch_pln3_pkd3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      unsigned int *x_offset_r,
                                      unsigned int *y_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *y_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_b,
                                      RpptROIPtr roiTensorPtrSrc)
{

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint src_pix_idx, dst_pix_idx;

    src_pix_idx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    dst_pix_idx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    dstPtr[dst_pix_idx] = srcPtr[src_pix_idx];
    dstPtr[dst_pix_idx + 1] = srcPtr[src_pix_idx + srcStridesNCH.y];
    dstPtr[dst_pix_idx + 2] = srcPtr[src_pix_idx + 2 * srcStridesNCH.y];

    int x_r, x_g, x_b, y_r, y_g, y_b;

    // R
    x_r = (id_x + x_offset_r[id_z]);
    y_r = (id_y + y_offset_r[id_z]);

    // G
    x_g = (id_x + x_offset_g[id_z]);
    y_g = (id_y + y_offset_g[id_z]);

    // B
    x_b = (id_x + x_offset_b[id_z]);
    y_b = (id_y + y_offset_b[id_z]);

    // R
    if ((y_r >= 0) && (y_r <= roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        dstPtr[dst_pix_idx] = srcPtr[ (id_z * srcStridesNCH.x) + (x_r + y_r * srcStridesNCH.z)];
    }

    // G
    if ((y_g >= 0) && (y_g <= roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        dstPtr[dst_pix_idx + 1] = srcPtr[(id_z * srcStridesNCH.x) + (x_g + y_g * srcStridesNCH.z) + srcStridesNCH.y];
    }

    // B
    if ((y_b >= 0) && (y_b <= roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        dstPtr[dst_pix_idx + 2] = srcPtr[(id_z * srcStridesNCH.x) + (x_b + y_b * srcStridesNCH.z) + 2 * srcStridesNCH.y];
    }
}


template <typename T>
RppStatus hip_exec_glitch_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pln3_pkd3_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pkd3_pln3_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}