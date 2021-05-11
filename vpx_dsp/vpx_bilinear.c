//
// Created by hyunho on 7/24/19.
//

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include "./vpx_bilinear.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vp9/common/vp9_onyxc_int.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static void vpx_bilinear_interp_horiz_uint8_c(const uint8_t *src, ptrdiff_t src_stride, int16_t *dst,  ptrdiff_t dst_stride, int width, int height, int scale, const nemo_bilinear_coeff_t *config){
    int x, y;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width * scale; x = x + 2) {
            const int left_x_index_0 = config->left_x_index[x];
            const int left_x_index_1 = config->left_x_index[x + 1];
//            const int right_x_index_0 = config->right_x_index[x];
//            const int right_x_index_1 = config->right_x_index[x + 1];
            const int right_x_index_0 = MIN(config->right_x_index[x], width - 1);
            const int right_x_index_1 = MIN(config->right_x_index[x + 1], width - 1);

            const int16_t x_lerp_fixed_0 = config->x_lerp_fixed[x];
            const int16_t x_lerp_fixed_1 = config->x_lerp_fixed[x + 1];

            const int16_t left_0 = src[y * src_stride + left_x_index_0];
            const int16_t right_0 = src[y * src_stride + right_x_index_0];
            const int16_t left_1 = src[y * src_stride + left_x_index_1];
            const int16_t right_1 = src[y * src_stride + right_x_index_1];

            const int16_t result_0 = left_0 + (((right_0 - left_0) * x_lerp_fixed_0 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
            const int16_t result_1 = left_1 + (((right_1 - left_1) * x_lerp_fixed_1 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);

            dst[y * dst_stride + x] = result_0;
            dst[y * dst_stride + (x + 1)] = result_1;
        }
    }
}

static void vpx_bilinear_interp_horiz_int16_c(const int16_t *src, ptrdiff_t src_stride, int16_t *dst,  ptrdiff_t dst_stride, int width, int height, int scale, const nemo_bilinear_coeff_t *config){
    int x, y;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width * scale; x = x + 2) {
            const int left_x_index_0 = config->left_x_index[x];
            const int left_x_index_1 = config->left_x_index[x + 1];
//            const int right_x_index_0 = config->right_x_index[x];
//            const int right_x_index_1 = config->right_x_index[x + 1];
            const int right_x_index_0 = MIN(config->right_x_index[x], width - 1);
            const int right_x_index_1 = MIN(config->right_x_index[x + 1], width - 1);

            const int16_t x_lerp_fixed_0 = config->x_lerp_fixed[x];
            const int16_t x_lerp_fixed_1 = config->x_lerp_fixed[x + 1];

            const int16_t left_0 = src[y * src_stride + left_x_index_0];
            const int16_t right_0 = src[y * src_stride + right_x_index_0];
            const int16_t left_1 = src[y * src_stride + left_x_index_1];
            const int16_t right_1 = src[y * src_stride + right_x_index_1];

            const int16_t result_0 = left_0 + (((right_0 - left_0) * x_lerp_fixed_0 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
            const int16_t result_1 = left_1 + (((right_1 - left_1) * x_lerp_fixed_1 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);

            dst[y * dst_stride + x] = result_0;
            dst[y * dst_stride + (x + 1)] = result_1;
        }
    }
}

static void vpx_bilinear_interp_vert_uint8_c(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,  ptrdiff_t dst_stride, int width, int height, int scale, const nemo_bilinear_coeff_t *config){
    int x, y;

    for (y = 0; y < height * scale; ++y) {
        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
        const int bottom_y_index = MIN(config->bottom_y_index[y], height - 1);
        const int16_t y_lerp_fixed = config->y_lerp_fixed[y];

        for (x = 0; x < width * scale; x = x + 2) {
            const int16_t top_0 = src[top_y_index * src_stride + x];
            const int16_t bottom_0 = src[bottom_y_index * src_stride + x];
            const int16_t top_1 = src[top_y_index * src_stride + (x + 1)];
            const int16_t bottom_1 = src[bottom_y_index * src_stride + (x + 1)];

            const int16_t result_0 = top_0 + (((bottom_0 - top_0) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
            const int16_t result_1 = top_1 + (((bottom_1 - top_1) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);

            dst[y * dst_stride + x] = result_0;
            dst[y * dst_stride + (x + 1)] = result_1;
        }
    }
}

static void vpx_bilinear_interp_vert_int16_c(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,  ptrdiff_t dst_stride, int width, int height, int scale, const nemo_bilinear_coeff_t *config){
    int x, y;

    for (y = 0; y < height * scale; ++y) {
        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
        const int bottom_y_index = MIN(config->bottom_y_index[y], height - 1);
        const int16_t y_lerp_fixed = config->y_lerp_fixed[y];

        for (x = 0; x < width * scale; x = x + 2) {
            const int16_t top_0 = src[top_y_index * src_stride + x];
            const int16_t bottom_0 = src[bottom_y_index * src_stride + x];
            const int16_t top_1 = src[top_y_index * src_stride + (x + 1)];
            const int16_t bottom_1 = src[bottom_y_index * src_stride + (x + 1)];

            const int16_t result_0 = top_0 + (((bottom_0 - top_0) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
            const int16_t result_1 = top_1 + (((bottom_1 - top_1) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);

            dst[y * dst_stride + x] = clip_pixel(dst[y * dst_stride + x] + result_0);
            dst[y * dst_stride + (x + 1)] = clip_pixel(dst[y * dst_stride + (x + 1)] + result_1);
        }
    }
}

void vpx_bilinear_interp_uint8_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,  ptrdiff_t dst_stride, int x_offset, int y_offset, int width, int height, int scale, const nemo_bilinear_coeff_t *config){
    int16_t temp[256 * 256];

    assert(width <= 64);
    assert(height <= 64);
    assert(scale <= 4 && scale >= 2);

    src = src + (y_offset * src_stride + x_offset);
    dst = dst + (y_offset * dst_stride + x_offset) * scale;

    vpx_bilinear_interp_horiz_uint8_c(src, src_stride, temp, 256, width, height, scale, config);
    vpx_bilinear_interp_vert_uint8_c(temp, 256, dst, dst_stride, width, height, scale, config);
}

void vpx_bilinear_interp_int16_c(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,  ptrdiff_t dst_stride, int x_offset, int y_offset, int width, int height, int scale, const nemo_bilinear_coeff_t *config){
    int16_t temp[256 * 256];

    assert(width <= 64);
    assert(height <= 64);
    assert(scale <= 4 && scale >= 2);

    src = src + (y_offset * src_stride + x_offset);
    dst = dst + (y_offset * dst_stride + x_offset) * scale;

    vpx_bilinear_interp_horiz_int16_c(src, src_stride, temp, 256, width, height, scale, config);
    vpx_bilinear_interp_vert_int16_c(temp, 256, dst, dst_stride, width, height, scale, config);
}
