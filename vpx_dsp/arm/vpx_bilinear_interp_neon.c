//
// Created by hyunho on 8/17/19.
// reference: ARM ne10 - void ne10_img_vresize_linear_neon (const int** src, unsigned char* dst, const short* beta, int width)
//
#include <arm_neon.h>
#include <assert.h>
#include <memory.h>
#include <sys/param.h>

#include "../vpx_bilinear.h"
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"
#include "../../vp9/common/vp9_onyxc_int.h"
#include "../vpx_dsp_common.h"

#define TAG "vpx_bilinear_interp_neon.c"
#define _UNKNOWN   0
#define _DEFAULT   1
#define _VERBOSE   2
#define _DEBUG    3
#define _INFO        4
#define _WARN        5
#define _ERROR    6
#define _FATAL    7
#define _SILENT       8

static void vpx_bilinear_interp_horiz_c_uint8(const uint8_t *src, ptrdiff_t src_stride,
                                              int16_t *dst, ptrdiff_t dst_stride, int width,
                                              int height, int scale,
                                              const nemo_bilinear_coeff_t *config) {
    int x, y;

    /*
    if (scale == 2 || scale == 3) {
        fprintf(stderr, "%s: Need to compare neon and c implementations", __func__);
        assert(1);
    }
    */

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

            const int16_t result_0 = left_0 + (((right_0 - left_0) * x_lerp_fixed_0 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT); //fixed-point
            const int16_t result_1 = left_1 + (((right_1 - left_1) * x_lerp_fixed_1 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT); //fixed-point

            dst[y * dst_stride + x] = result_0;
            dst[y * dst_stride + (x + 1)] = result_1;
        }
    }
}

static void vpx_bilinear_interp_horiz_c_int16(const int16_t *src, ptrdiff_t src_stride,
                                              int16_t *dst, ptrdiff_t dst_stride, int width,
                                              int height, int scale,
                                              const nemo_bilinear_coeff_t *config) {
    int x, y;

    /*
    if (scale == 2 || scale == 3) {
        fprintf(stderr, "%s: Need to compare neon and c implementations", __func__);
        assert(1);
    }
    */

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

            const int16_t result_0 =
                    left_0 + (((right_0 - left_0) * x_lerp_fixed_0 + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT); //fixed-point
            const int16_t result_1 = left_1 + (((right_1 - left_1) * x_lerp_fixed_1 + BILINEAR_DELTA)
                    >> BILINEAR_FRACTION_BIT); //fixed-point

            dst[y * dst_stride + x] = result_0;
            dst[y * dst_stride + (x + 1)] = result_1;
        }
    }
}

static void
vpx_bilinear_interp_vert_neon_w8_uint8(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,
                                       ptrdiff_t dst_stride, int width, int height, int scale,
                                       const nemo_bilinear_coeff_t *config) {
    int x, y;

    int16x8_t qS0_01234567_0, qS1_01234567_0;
    int16x8_t qT_01234567_0;
    int16x8_t y_lerp_fixed_v;

    //fprintf(stderr, "%s: Not tested yet", __func__);
    //assert(1);

    for (y = 0; y < height * scale; ++y) {
        x = 0;
        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
        const int bottom_y_index = MIN(config->bottom_y_index[y], height - 1);
        const int16_t y_lerp_fixed = config->y_lerp_fixed[y];
        y_lerp_fixed_v = vdupq_n_s16(y_lerp_fixed);

        //1. process 8 values * 2 loop unrolling == 16
        for (; x <= width * scale - 8; x += 8) {
            //a. load source pixels: top0,1, bottom0,1
            qS0_01234567_0 = vld1q_s16(&src[top_y_index * src_stride + x]);
            qS1_01234567_0 = vld1q_s16(&src[bottom_y_index * src_stride + x]);

            //b. interpolate pixels
            qT_01234567_0 = vsubq_s16(qS1_01234567_0, qS0_01234567_0);
            qT_01234567_0 = vmulq_s16(qT_01234567_0, y_lerp_fixed_v);
            qT_01234567_0 = vrsraq_n_s16(qS0_01234567_0, qT_01234567_0, BILINEAR_FRACTION_BIT);

            //c. save pixels
            vst1_u8(&dst[y * dst_stride + x], vmovn_u16(vreinterpretq_u16_s16(qT_01234567_0)));
        }

        //2. process remaining 8 values // case: 12x12
        for (x = 0; x < width * scale; x = x + 2) {
            //fprintf(stderr, "%s: Not tested yet", __func__);
            //assert(1);

            const int16_t top_0 = src[top_y_index * src_stride + x];
            const int16_t bottom_0 = src[bottom_y_index * src_stride + x];
            const int16_t top_1 = src[top_y_index * src_stride + (x + 1)];
            const int16_t bottom_1 = src[bottom_y_index * src_stride + (x + 1)];

            const int16_t result_0 = top_0 + (((bottom_0 - top_0) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
            const int16_t result_1 = top_1 + (((bottom_1 - top_1) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);

            dst[y * dst_stride + x] = clip_pixel(result_0);
            dst[y * dst_stride + (x + 1)] = clip_pixel(result_1);
        }
    }


//    int x, y;
//
//    for (y = 0; y < height * scale; ++y) {
//        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
//        const float y_lerp = config->y_lerp[y];
////        const int16_t y_lerp_fixed = config->y_lerp_fixed[y];
//
//        for (x = 0; x < width * scale; x = x + 2) {
//            const int16_t top_0 = src[top_y_index * src_stride + x];
//            const int16_t bottom_0 = src[bottom_y_index * src_stride + x];
//            const int16_t top_1 = src[top_y_index * src_stride + (x + 1)];
//            const int16_t bottom_1 = src[bottom_y_index * src_stride + (x + 1)];
//
//            const int16_t result_0 = top_0 + ((bottom_0 - top_0) * y_lerp);
//            const int16_t result_1 = top_1 + ((bottom_1 - top_1) * y_lerp);
////            const int16_t result_0 = top_0 + (((bottom_0 - top_0) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
////            const int16_t result_1 = top_1 + (((bottom_1 - top_1) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
//
//            dst[y * dst_stride + x] = clip_pixel(result_0);
//            dst[y * dst_stride + (x + 1)] = clip_pixel(result_1);
//        }
//    }
}


static void
vpx_bilinear_interp_vert_neon_w16_uint8(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,
                                        ptrdiff_t dst_stride, int width, int height, int scale,
                                        const nemo_bilinear_coeff_t *config) {
    int x, y;

    int16x8_t qS0_01234567_0, qS0_01234567_1, qS1_01234567_0, qS1_01234567_1;
    int16x8_t qT_01234567_0, qT_01234567_1;
    uint8x8_t dDst_01234567_0;
    uint16x8_t dT0_01234567_0;
    int16x8_t y_lerp_fixed_v;

    for (y = 0; y < height * scale; ++y) {
        x = 0;
        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
        const int bottom_y_index = MIN(config->bottom_y_index[y], height - 1);
        y_lerp_fixed_v = vdupq_n_s16(config->y_lerp_fixed[y]);

        //1. process 8 values * 2 loop unrolling == 16
        for (; x <= width * scale - 16; x += 16) {
            //a. load source pixels: top0,1, bottom0,1
            qS0_01234567_0 = vld1q_s16(&src[top_y_index * src_stride + x]);
            qS0_01234567_1 = vld1q_s16(&src[top_y_index * src_stride + x + 8]);
            qS1_01234567_0 = vld1q_s16(&src[bottom_y_index * src_stride + x]);
            qS1_01234567_1 = vld1q_s16(&src[bottom_y_index * src_stride + x + 8]);

            //b. interpolate pixels
            qT_01234567_0 = vsubq_s16(qS1_01234567_0, qS0_01234567_0);
            qT_01234567_1 = vsubq_s16(qS1_01234567_1, qS0_01234567_1);
            qT_01234567_0 = vmulq_s16(qT_01234567_0, y_lerp_fixed_v);
            qT_01234567_1 = vmulq_s16(qT_01234567_1, y_lerp_fixed_v);
            qT_01234567_0 = vrsraq_n_s16(qS0_01234567_0, qT_01234567_0, BILINEAR_FRACTION_BIT);
            qT_01234567_1 = vrsraq_n_s16(qS0_01234567_1, qT_01234567_1, BILINEAR_FRACTION_BIT);

            //c. save pixels
            vst1_u8(&dst[y * dst_stride + x], vmovn_u16(vreinterpretq_u16_s16(qT_01234567_0)));
            vst1_u8(&dst[y * dst_stride + x + 8], vmovn_u16(vreinterpretq_u16_s16(qT_01234567_1)));
        }

        //2. process remaining 8 values // case: 24x24
        for (; x < width * scale; x += 8) {
            //fprintf(stderr, "%s: Not tested yet", __func__);
            //assert(1);

            //a. load source pixels: top0,1, bottom0,1
            qS0_01234567_0 = vld1q_s16(&src[top_y_index * src_stride + x]);
            qS1_01234567_0 = vld1q_s16(&src[bottom_y_index * src_stride + x]);

            //b. interpolate pixels
            qT_01234567_0 = vsubq_s16(qS1_01234567_0, qS0_01234567_0);
            qT_01234567_0 = vmulq_s16(qT_01234567_0, y_lerp_fixed_v);
            qT_01234567_0 = vrsraq_n_s16(qS0_01234567_0, qT_01234567_0, BILINEAR_FRACTION_BIT);

            //c. save pixels
            vst1_u8(&dst[y * dst_stride + x], vmovn_u16(vreinterpretq_u16_s16(qT_01234567_0)));
        }
    }
}

static void
vpx_bilinear_interp_vert_neon_w8_int16(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,
                                       ptrdiff_t dst_stride, int width, int height, int scale,
                                       const nemo_bilinear_coeff_t *config) {
    //fprintf(stderr, "%s: Not tested yet", __func__);
    //assert(1);

    int x, y;

    int16x8_t qS0_01234567_0, qS1_01234567_0;
    int16x8_t qT_01234567_0;
    uint8x8_t dDst_01234567_0;
    uint16x8_t dT0_01234567_0;
    int16x8_t y_lerp_fixed_v;
    for (y = 0; y < height * scale; ++y) {
        x = 0;
        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
        const int bottom_y_index = MIN(config->bottom_y_index[y], height - 1);
        const int16_t y_lerp_fixed = config->y_lerp_fixed[y];
        y_lerp_fixed_v = vdupq_n_s16(y_lerp_fixed);

        //1. process 8 values * 2 loop unrolling == 16 // case: 8x8, 12x12
        for (; x <= width * scale - 16; x += 16) {
            //a. load source pixels: top0,1, bottom0,1
            qS0_01234567_0 = vld1q_s16(&src[top_y_index * src_stride + x]);
            qS1_01234567_0 = vld1q_s16(&src[bottom_y_index * src_stride + x]);

            //b. interpolate pixels
            qT_01234567_0 = vsubq_s16(qS1_01234567_0, qS0_01234567_0);
            qT_01234567_0 = vmulq_s16(qT_01234567_0, y_lerp_fixed_v);
            qT_01234567_0 = vrsraq_n_s16(qS0_01234567_0, qT_01234567_0, BILINEAR_FRACTION_BIT);

            //c. load & add destination pixels
            //d. clip pixels
            //e. convert from in16 to uint8
            dDst_01234567_0 = vld1_u8(&dst[y * dst_stride + x]);
            dT0_01234567_0 = vaddw_u8(vreinterpretq_u16_s16(qT_01234567_0), dDst_01234567_0);
            dDst_01234567_0 = vqmovun_s16(vreinterpretq_s16_u16(dT0_01234567_0));

            //f. save pixels
            vst1_u8(&dst[y * dst_stride + x], dDst_01234567_0);
        }

        //2. process remaining 4 values // case: 12x12
        for (; x < width * scale; x += 2) { //hyunho: it consists of only 4 pixels, so process by non-neon instructions
            //fprintf(stderr, "%s: Not tested yet", __func__);
            //assert(1);

            //a. load source pixels: top0,1, bottom0,1
            const int16_t top_0 = src[top_y_index * src_stride + x];
            const int16_t bottom_0 = src[bottom_y_index * src_stride + x];
            const int16_t top_1 = src[top_y_index * src_stride + (x + 1)];
            const int16_t bottom_1 = src[bottom_y_index * src_stride + (x + 1)];

            //b. interpolate pixels
            const int16_t result_0 =
                    top_0 + (((bottom_0 - top_0) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);
            const int16_t result_1 =
                    top_1 + (((bottom_1 - top_1) * y_lerp_fixed + BILINEAR_DELTA) >> BILINEAR_FRACTION_BIT);

            //c. load & add destination pixels
            //d. clip pixels
            //e. convert from in16 to uint8
            //f. save pixels
            dst[y * dst_stride + x] = clip_pixel(dst[y * dst_stride + x] + result_0);
            dst[y * dst_stride + (x + 1)] = clip_pixel(dst[y * dst_stride + (x + 1)] + result_1);
        }
    }
}

static void
vpx_bilinear_interp_vert_neon_w16_int16(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,
                                        ptrdiff_t dst_stride, int width, int height, int scale,
                                        const nemo_bilinear_coeff_t *config) {
    int x, y;

    int16x8_t qS0_01234567_0, qS0_01234567_1, qS1_01234567_0, qS1_01234567_1;
    int16x8_t qT_01234567_0, qT_01234567_1;
    uint8x8_t dDst_01234567_0, dDst_01234567_1;
    uint16x8_t dT0_01234567_0, dT0_01234567_1;
    int16x8_t y_lerp_fixed_v;

    for (y = 0; y < height * scale; ++y) {
        x = 0;
        const int top_y_index = config->top_y_index[y];
//        const int bottom_y_index = config->bottom_y_index[y];
        const int bottom_y_index = MIN(config->bottom_y_index[y], height - 1);
        y_lerp_fixed_v = vdupq_n_s16(config->y_lerp_fixed[y]);

        //1. process 8 values * 2 loop unrolling == 16
        for (; x <= width * scale - 16; x += 16) {
            //a. load source pixels: top0,1, bottom0,1
            qS0_01234567_0 = vld1q_s16(&src[top_y_index * src_stride + x]);
            qS0_01234567_1 = vld1q_s16(&src[top_y_index * src_stride + x + 8]);
            qS1_01234567_0 = vld1q_s16(&src[bottom_y_index * src_stride + x]);
            qS1_01234567_1 = vld1q_s16(&src[bottom_y_index * src_stride + x + 8]);

            //b. interpolate pixels
            qT_01234567_0 = vsubq_s16(qS1_01234567_0, qS0_01234567_0);
            qT_01234567_1 = vsubq_s16(qS1_01234567_1, qS0_01234567_1);
            qT_01234567_0 = vmulq_s16(qT_01234567_0, y_lerp_fixed_v);
            qT_01234567_1 = vmulq_s16(qT_01234567_1, y_lerp_fixed_v);
            qT_01234567_0 = vrsraq_n_s16(qS0_01234567_0, qT_01234567_0, BILINEAR_FRACTION_BIT);
            qT_01234567_1 = vrsraq_n_s16(qS0_01234567_1, qT_01234567_1, BILINEAR_FRACTION_BIT);

            //c. load & add destination pixels
            //d. clip pixels
            //e. convert from in16 to uint8
            dDst_01234567_0 = vld1_u8(&dst[y * dst_stride + x]);
            dDst_01234567_1 = vld1_u8(&dst[y * dst_stride + x + 8]);
            dT0_01234567_0 = vaddw_u8(vreinterpretq_u16_s16(qT_01234567_0), dDst_01234567_0);
            dT0_01234567_1 = vaddw_u8(vreinterpretq_u16_s16(qT_01234567_1), dDst_01234567_1);
            dDst_01234567_0 = vqmovun_s16(vreinterpretq_s16_u16(dT0_01234567_0));
            dDst_01234567_1 = vqmovun_s16(vreinterpretq_s16_u16(dT0_01234567_1));

            //f. save pixels
            vst1_u8(&dst[y * dst_stride + x], dDst_01234567_0);
            vst1_u8(&dst[y * dst_stride + x + 8], dDst_01234567_1);
        }

        //2. process remaining 8 values // case: 24x24
        //TODO (hyunho) - need quality & latency test for scale x3
        for (; x < width * scale; x += 8) {
            //fprintf(stderr, "Not tested yet: vpx_bilinear_interp_vert_neon_w16_int16()");
            //assert(1);

            //a. load source pixels: top0,1, bottom0,1
            qS0_01234567_0 = vld1q_s16(&src[top_y_index * src_stride + x]);
            qS1_01234567_0 = vld1q_s16(&src[bottom_y_index * src_stride + x]);

            //b. interpolate pixels
            qT_01234567_0 = vsubq_s16(qS1_01234567_0, qS0_01234567_0);
            qT_01234567_0 = vmulq_s16(qT_01234567_0, y_lerp_fixed_v);
            qT_01234567_0 = vrsraq_n_s16(qS0_01234567_0, qT_01234567_0, BILINEAR_FRACTION_BIT);

            //c. load & add destination pixels
            //d. clip pixels
            //e. convert from in16 to uint8
            dDst_01234567_0 = vld1_u8(&dst[y * dst_stride + x]);
            dT0_01234567_0 = vaddw_u8(vreinterpretq_u16_s16(qT_01234567_0), dDst_01234567_0);
            dDst_01234567_0 = vqmovun_s16(vreinterpretq_s16_u16(dT0_01234567_0));

            //f. save pixels
            vst1_u8(&dst[y * dst_stride + x], dDst_01234567_0);
        }
    }
}

void vpx_bilinear_interp_int16_neon(const int16_t *src, ptrdiff_t src_stride, uint8_t *dst,
                                    ptrdiff_t dst_stride, int x_offset, int y_offset, int width,
                                    int height, int scale, const nemo_bilinear_coeff_t *config) {
    int16_t temp[256 * 256];
    int h = height * scale;

    assert(width <= 64);
    assert(height <= 64);
    assert(scale <= 4 && scale >= 2);

    src = src + (y_offset * src_stride + x_offset);
    dst = dst + (y_offset * dst_stride + x_offset) * scale;

    vpx_bilinear_interp_horiz_c_int16(src, src_stride, temp, 256, width, height, scale, config);
    if (h >= 16) {
        vpx_bilinear_interp_vert_neon_w16_int16(temp, 256, dst, dst_stride, width, height, scale,
                                                config);
    } else {
        vpx_bilinear_interp_vert_neon_w8_int16(temp, 256, dst, dst_stride, width, height, scale,
                                               config);
    }
}

void vpx_bilinear_interp_uint8_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                                    ptrdiff_t dst_stride, int x_offset, int y_offset, int width,
                                    int height, int scale, const nemo_bilinear_coeff_t *config) {
    int16_t temp[128 * 128];
    int h = height * scale;

    assert(width <= 64);
    assert(height <= 64);
    assert(scale <= 4 && scale >= 2);

    src = src + (y_offset * src_stride + x_offset);
    dst = dst + (y_offset * dst_stride + x_offset) * scale;

    vpx_bilinear_interp_horiz_c_uint8(src, src_stride, temp, 256, width, height, scale, config);
    if (h >= 16) {
        vpx_bilinear_interp_vert_neon_w16_uint8(temp, 256, dst, dst_stride, width, height, scale,
                                                config);
    } else {
        vpx_bilinear_interp_vert_neon_w8_uint8(temp, 256, dst, dst_stride, width, height, scale,
                                                config);
    }
}
