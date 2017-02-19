/*
Copyright 2016 fixstars

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//#include <nppi.h>

#include "internal.h"



//#include "internal.h"
#include <inttypes.h>

extern "C"
{

    // clamp condition
    __device__ inline int clampBC(const int x, const int y, const int nx, const int ny)
    {
        const int idx = min(max(x, 0), nx - 1);
        const int idy = min(max(y, 0), ny - 1);
        return idx + idy * nx;
    }

    __global__ void median_kernel_3x3(
        const float* input, float* output, int nx, int ny)
    {
        const int idy = threadIdx.y + blockIdx.y * blockDim.y;
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;

        const int id = idx + idy * nx;

        if (idx >= nx || idy >= ny)
            return;

        float window[9];

        window[0] = input[clampBC(idx - 1, idy - 1, nx, ny)];
        window[1] = input[clampBC(idx, idy - 1, nx, ny)];
        window[2] = input[clampBC(idx + 1, idy - 1, nx, ny)];

        window[3] = input[clampBC(idx - 1, idy, nx, ny)];
        window[4] = input[clampBC(idx, idy, nx, ny)];
        window[5] = input[clampBC(idx + 1, idy, nx, ny)];

        window[6] = input[clampBC(idx - 1, idy + 1, nx, ny)];
        window[7] = input[clampBC(idx, idy + 1, nx, ny)];
        window[8] = input[clampBC(idx + 1, idy + 1, nx, ny)];

        // perform partial bitonic sort to find current median
        float flMin = min(window[0], window[1]);
        float flMax = max(window[0], window[1]);
        window[0] = flMin;
        window[1] = flMax;

        flMin = min(window[3], window[2]);
        flMax = max(window[3], window[2]);
        window[3] = flMin;
        window[2] = flMax;

        flMin = min(window[2], window[0]);
        flMax = max(window[2], window[0]);
        window[2] = flMin;
        window[0] = flMax;

        flMin = min(window[3], window[1]);
        flMax = max(window[3], window[1]);
        window[3] = flMin;
        window[1] = flMax;

        flMin = min(window[1], window[0]);
        flMax = max(window[1], window[0]);
        window[1] = flMin;
        window[0] = flMax;

        flMin = min(window[3], window[2]);
        flMax = max(window[3], window[2]);
        window[3] = flMin;
        window[2] = flMax;

        flMin = min(window[5], window[4]);
        flMax = max(window[5], window[4]);
        window[5] = flMin;
        window[4] = flMax;

        flMin = min(window[7], window[8]);
        flMax = max(window[7], window[8]);
        window[7] = flMin;
        window[8] = flMax;

        flMin = min(window[6], window[8]);
        flMax = max(window[6], window[8]);
        window[6] = flMin;
        window[8] = flMax;

        flMin = min(window[6], window[7]);
        flMax = max(window[6], window[7]);
        window[6] = flMin;
        window[7] = flMax;

        flMin = min(window[4], window[8]);
        flMax = max(window[4], window[8]);
        window[4] = flMin;
        window[8] = flMax;

        flMin = min(window[4], window[6]);
        flMax = max(window[4], window[6]);
        window[4] = flMin;
        window[6] = flMax;

        flMin = min(window[5], window[7]);
        flMax = max(window[5], window[7]);
        window[5] = flMin;
        window[7] = flMax;

        flMin = min(window[4], window[5]);
        flMax = max(window[4], window[5]);
        window[4] = flMin;
        window[5] = flMax;

        flMin = min(window[6], window[7]);
        flMax = max(window[6], window[7]);
        window[6] = flMin;
        window[7] = flMax;

        flMin = min(window[0], window[8]);
        flMax = max(window[0], window[8]);
        window[0] = flMin;
        window[8] = flMax;

        window[4] = max(window[0], window[4]);
        window[5] = max(window[1], window[5]);

        window[6] = max(window[2], window[6]);
        window[7] = max(window[3], window[7]);

        window[4] = min(window[4], window[6]);
        window[5] = min(window[5], window[7]);

        output[id] = min(window[4], window[5]);
    }

}



namespace sgm {
	namespace details {

		void median_filter(const float* d_src, float* d_dst, void* median_filter_buffer, int width, int height) {
			//NppiSize roi = { width, height };
			//NppiSize mask = { 3, 3 };
			//NppiPoint anchor = { 0, 0 };
            //
			//NppStatus status = nppiFilterMedian_32f_C1R(d_src, sizeof(Npp32f) * width, d_dst, sizeof(Npp32f) * width, roi, mask, anchor, (Npp8u*)median_filter_buffer);
			//
			//assert(status == 0);
            const dim3 blocks(width / 16, height / 16);
            const dim3 threads(16, 16);
            median_kernel_3x3 << < blocks, threads >> > (d_src, d_dst, width, height);
            CudaKernelCheck();
		}

	}
}
