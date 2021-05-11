////
//// Created by hyunho on 6/30/20.
////

#include <stdint.h>
#include <arm_neon.h>
#include <math.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_nemo.h"
#include "../../vpx/vpx_image.h"
#include "../../vpx_dsp/vpx_dsp_common.h"
#include "../../vpx_scale/yv12config.h"
#include "../../vpx/vpx_nemo.h"
#include "../vpx_convert.h"

//TODO: move to vpx_dsp
int RGB24_float_to_uint8_neon(RGB24_BUFFER_CONFIG *rbf) {
    if (rbf == NULL) {
        return -1;
    }

    float *src = rbf->buffer_alloc_float;
    uint8_t *dst = rbf->buffer_alloc;

    int w, h;

    const float init[4] = {0.5, 0.5, 0.5, 0.5};
    float32x4_t c0 = vld1q_f32(init);

    for (h = 0; h < rbf->height; h++) {
        for (w = 0; w <= rbf->width - 8; w += 8) {
            float32x4_t src_float_0 = vld1q_f32(src + w + h * rbf->stride);
            src_float_0 = vaddq_f32(src_float_0, c0);
            uint32x4_t dst_uint32_0 = vcvtq_u32_f32(src_float_0);
            uint16x4_t dst_uint16_0 = vqmovn_s32(dst_uint32_0);

            float32x4_t src_float_1 = vld1q_f32(src + w + 4 + h * rbf->stride);
            src_float_1 = vaddq_f32(src_float_1, c0);
            uint32x4_t dst_uint32_1 = vcvtq_u32_f32(src_float_1);
            uint16x4_t dst_uint16_1 = vqmovn_s32(dst_uint32_1);

            uint16x8_t dst_uint16 = vcombine_u16(dst_uint16_0, dst_uint16_1);
            uint8x8_t dst_uint8 = vqmovn_u16(dst_uint16);

            vst1_u8(dst + w + h * rbf->stride, dst_uint8);
        }

        for (; w < rbf->width; w++) {
            *(dst + w + h * rbf->stride) = clamp(round(*(src + w + h * rbf->stride)), 0, 255);
        }
    }
    return 0;
}

void RGB24_to_YV12_bt701_neon(YV12_BUFFER_CONFIG *ybf, RGB24_BUFFER_CONFIG *rbf) {
    uint8_t r, g, b;
    uint8x8_t ry_coeff = vdup_n_u8(RY_COEFF_INT);
    uint8x8_t gy_coeff = vdup_n_u8(GY_COEFF_INT);
    uint8x8_t by_coeff = vdup_n_u8(BY_COEFF_INT);
    uint8x8_t ru_coeff = vdup_n_u8(RU_COEFF_INT);
    uint8x8_t gu_coeff = vdup_n_u8(GU_COEFF_INT);
    uint8x8_t bu_coeff = vdup_n_u8(BU_COEFF_INT);
    uint8x8_t rv_coeff = vdup_n_u8(RV_COEFF_INT);
    uint8x8_t gv_coeff = vdup_n_u8(GV_COEFF_INT);
    uint8x8_t bv_coeff = vdup_n_u8(BV_COEFF_INT);
    uint8x8_t y_offset = vdup_n_u8(Y_OFFSET);
    uint8x8_t u_offset = vdup_n_u8(U_OFFSET);
    uint8x8_t v_offset = vdup_n_u8(V_OFFSET);
    uint8x8x3_t rgb_n, rgb1_n;
    uint16x8_t y_w, y1_w, u_w, u1_w, v_w, v1_w;
    uint8x8_t y_n, y1_n, u_n, u1_n, v_n, v1_n;

    int i, j;
    const int height = ybf->y_crop_height;
    const int width = ybf->y_crop_width;
    for (i = 0; i < height; i++) {
        for (j = 0; j <= width - 16; j += 16) {
            rgb_n = vld3_u8(rbf->buffer_alloc + i * rbf->stride + j * 3);
            rgb1_n = vld3_u8(rbf->buffer_alloc + i * rbf->stride + (j + 8) * 3);

            y_w = vmlal_u8(y_w, ry_coeff, rgb_n.val[0]);
            y1_w = vmlal_u8(y1_w, ry_coeff, rgb1_n.val[0]);
            y_w = vmlal_u8(y_w, gy_coeff, rgb_n.val[1]);
            y1_w = vmlal_u8(y1_w, gy_coeff, rgb1_n.val[1]);
            y_w = vmlal_u8(y_w, by_coeff, rgb_n.val[2]);
            y1_w = vmlal_u8(y1_w, by_coeff, rgb1_n.val[2]);

            y_w = vrshrq_n_u16(y_w, CONVERT_FRACTION_BIT);
            y1_w = vrshrq_n_u16(y1_w, CONVERT_FRACTION_BIT);

            y_n = vmovn_u16(y_w);
            y1_n = vmovn_u16(y1_w);

            y_n = vadd_u8(y_n, y_offset);
            y1_n = vadd_u8(y1_n, y_offset);

            vst1_u8(ybf->y_buffer + i * ybf->y_stride + j, y_n);
            vst1_u8(ybf->y_buffer + i * ybf->y_stride + j + 8, y1_n);
        }

        for (; j < width; j++) {
            r = *(rbf->buffer_alloc + i * rbf->stride + j * 3);
            g = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1);
            b = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2);
            *(ybf->y_buffer + i * ybf->y_stride + j) = (uint8_t) (((RY_COEFF_INT * r + GY_COEFF_INT * g + BY_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) +
                                                                  Y_OFFSET);
        }
    }

    for (i = 0; i < height; i = i + (1 << ybf->subsampling_y)) {
        for (j = 0; j <= width - (16 << ybf->subsampling_x); j = j + (16 << ybf->subsampling_x)) {
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + j * 3, rgb_n, 0);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (1 << ybf->subsampling_x)) * 3, rgb_n, 1);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (2 << ybf->subsampling_x)) * 3, rgb_n, 2);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (3 << ybf->subsampling_x)) * 3, rgb_n, 3);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (4 << ybf->subsampling_x)) * 3, rgb_n, 4);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (5 << ybf->subsampling_x)) * 3, rgb_n, 5);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (6 << ybf->subsampling_x)) * 3, rgb_n, 6);
            rgb_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (7 << ybf->subsampling_x)) * 3, rgb_n, 7);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (8 << ybf->subsampling_x)) * 3, rgb1_n, 0);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (9 << ybf->subsampling_x)) * 3, rgb1_n, 1);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (10 << ybf->subsampling_x)) * 3, rgb1_n, 2);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (11 << ybf->subsampling_x)) * 3, rgb1_n, 3);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (12 << ybf->subsampling_x)) * 3, rgb1_n, 4);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (13 << ybf->subsampling_x)) * 3, rgb1_n, 5);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (14 << ybf->subsampling_x)) * 3, rgb1_n, 6);
            rgb1_n = vld3_lane_u8(rbf->buffer_alloc + i * rbf->stride + (j + (15 << ybf->subsampling_x)) * 3, rgb1_n, 7);

            u_w = vmlal_u8(u_w, bu_coeff, rgb_n.val[2]);
            u_w = vmlsl_u8(u_w, ru_coeff, rgb_n.val[0]);
            u_w = vmlsl_u8(u_w, gu_coeff, rgb_n.val[1]);
            u1_w = vmlal_u8(u1_w, bu_coeff, rgb1_n.val[2]);
            u1_w = vmlsl_u8(u1_w, ru_coeff, rgb1_n.val[0]);
            u1_w = vmlsl_u8(u1_w, gu_coeff, rgb1_n.val[1]);

            v_w = vmlal_u8(v_w, rv_coeff, rgb_n.val[0]);
            v_w = vmlsl_u8(v_w, gv_coeff, rgb_n.val[1]);
            v_w = vmlsl_u8(v_w, bv_coeff, rgb_n.val[2]);
            v1_w = vmlal_u8(v1_w, rv_coeff, rgb1_n.val[0]);
            v1_w = vmlsl_u8(v1_w, gv_coeff, rgb1_n.val[1]);
            v1_w = vmlsl_u8(v1_w, bv_coeff, rgb1_n.val[2]);

            u_w = vrshrq_n_u16(u_w, CONVERT_FRACTION_BIT);
            v_w = vrshrq_n_u16(v_w, CONVERT_FRACTION_BIT);
            u1_w = vrshrq_n_u16(u1_w, CONVERT_FRACTION_BIT);
            v1_w = vrshrq_n_u16(v1_w, CONVERT_FRACTION_BIT);

            u_n = vmovn_u16(u_w);
            v_n = vmovn_u16(v_w);
            u1_n = vmovn_u16(u1_w);
            v1_n = vmovn_u16(v1_w);

            u_n = vadd_u8(u_n, u_offset);
            v_n = vadd_u8(v_n, v_offset);
            u1_n = vadd_u8(u1_n, u_offset);
            v1_n = vadd_u8(v1_n, v_offset);

            vst1_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x), u_n);
            vst1_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x), v_n);
            vst1_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 8, u1_n);
            vst1_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 8, v1_n);
        }

        for (; j < width; j++) {
            r = *(rbf->buffer_alloc + i * rbf->stride + j * 3);
            g = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1);
            b = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2);
            *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x)) = (uint8_t) (
                    ((-RU_COEFF_INT * r - GU_COEFF_INT * g + BU_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + U_OFFSET);
            *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x)) = (uint8_t) (
                    ((RV_COEFF_INT * r - GV_COEFF_INT * g - BV_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + V_OFFSET);
        }
    }
}

void YV12_to_RGB24_bt701_neon(RGB24_BUFFER_CONFIG *rbf, YV12_BUFFER_CONFIG *ybf) {
    uint8_t y, u, v;
    uint8x8_t yr_coeff = vdup_n_u8(YR_COEFF_INT);
    uint8x8_t ur_coeff = vdup_n_u8(UR_COEFF_INT);
    uint8x8_t vr_coeff = vdup_n_u8(VR_COEFF_INT);
    uint8x8_t yg_coeff = vdup_n_u8(YG_COEFF_INT);
    uint8x8_t ug_coeff = vdup_n_u8(UG_COEFF_INT);
    uint8x8_t vg_coeff = vdup_n_u8(VG_COEFF_INT);
    uint8x8_t yb_coeff = vdup_n_u8(YB_COEFF_INT);
    uint8x8_t ub_coeff = vdup_n_u8(UB_COEFF_INT);
    uint8x8_t vb_coeff = vdup_n_u8(VB_COEFF_INT);
    uint8x8_t y_offset = vdup_n_u8(Y_OFFSET);
    uint8x8_t u_offset = vdup_n_u8(U_OFFSET);
    uint8x8_t v_offset = vdup_n_u8(V_OFFSET);
    uint16x8_t r_offset = vdupq_n_u16(R_OFFSET);
    uint16x8_t g_offset = vdupq_n_u16(G_OFFSET);
    uint16x8_t b_offset = vdupq_n_u16(B_OFFSET);
    uint16x8_t max = vdupq_n_u16(255);
    uint16x8x3_t rgb_w;
    uint8x8x3_t rgb_n;
    uint8x8_t y_n, u_n, v_n;

    int i, j;
    const int height = ybf->y_crop_height;
    const int width = ybf->y_crop_width;
    for (i = 0; i < height; i++) {
        for (j = 0; j <= width - 8 ; j += 8) {
            rgb_w.val[0] = vdupq_n_u16(0);
            rgb_w.val[1] = vdupq_n_u16(0);
            rgb_w.val[2] = vdupq_n_u16(0);

            y_n = vld1_u8(ybf->y_buffer + i * ybf->y_stride + j);

            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x), u_n, 0);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 1) >> ybf->subsampling_x), u_n, 1);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 2) >> ybf->subsampling_x), u_n, 2);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 3) >> ybf->subsampling_x), u_n, 3);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 4) >> ybf->subsampling_x), u_n, 4);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 5) >> ybf->subsampling_x), u_n, 5);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 6) >> ybf->subsampling_x), u_n, 6);
            u_n = vld1_lane_u8(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 7) >> ybf->subsampling_x), u_n, 7);

            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x), v_n, 0);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 1) >> ybf->subsampling_x), v_n, 1);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 2) >> ybf->subsampling_x), v_n, 2);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 3) >> ybf->subsampling_x), v_n, 3);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 4) >> ybf->subsampling_x), v_n, 4);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 5) >> ybf->subsampling_x), v_n, 5);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 6) >> ybf->subsampling_x), v_n, 6);
            v_n = vld1_lane_u8(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 7) >> ybf->subsampling_x), v_n, 7);

            rgb_w.val[0] = vmlal_u8(rgb_w.val[0], yr_coeff, y_n);
            rgb_w.val[0] = vmlal_u8(rgb_w.val[0], vr_coeff, v_n);
            rgb_w.val[0] = vsubq_u16(rgb_w.val[0], r_offset);

            rgb_w.val[1] = vmlal_u8(rgb_w.val[1], yg_coeff, y_n);
            rgb_w.val[1] = vmlsl_u8(rgb_w.val[1], ug_coeff, u_n);
            rgb_w.val[1] = vmlsl_u8(rgb_w.val[1], vg_coeff, v_n);
            rgb_w.val[1] = vaddq_u16(rgb_w.val[1], g_offset);

            rgb_w.val[2] = vmlal_u8(rgb_w.val[2], yb_coeff, y_n);
            rgb_w.val[2] = vmlal_u8(rgb_w.val[2], ub_coeff, u_n);
            rgb_w.val[2] = vsubq_u16(rgb_w.val[2], b_offset);

            rgb_w.val[0] = vrshrq_n_u16(rgb_w.val[0], CONVERT_FRACTION_BIT);
            rgb_w.val[0] = vminq_u16(rgb_w.val[0], max);
            rgb_n.val[0] = vmovn_u16(rgb_w.val[0]);

            rgb_w.val[1] = vrshrq_n_u16(rgb_w.val[1], CONVERT_FRACTION_BIT);
            rgb_w.val[1] = vminq_u16(rgb_w.val[1], max);
            rgb_n.val[1] = vmovn_u16(rgb_w.val[1]);

            rgb_w.val[2] = vrshrq_n_u16(rgb_w.val[2], CONVERT_FRACTION_BIT);
            rgb_w.val[2] = vminq_u16(rgb_w.val[2], max);
            rgb_n.val[2] = vmovn_u16(rgb_w.val[2]);

            vst3_u8(rbf->buffer_alloc + i * rbf->stride + j * 3, rgb_n);
        }

        for (; j < width; j++) {
            y = *(ybf->y_buffer + i * ybf->y_stride + j);
            u = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));
            v = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));

            *(rbf->buffer_alloc + i * rbf->stride + j * 3) = (uint8_t)
                    clamp(((YR_COEFF_INT * (y - Y_OFFSET) + VR_COEFF_INT * (v - V_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // R value
            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1) = (uint8_t)
                    clamp(((YG_COEFF_INT * (y - Y_OFFSET) - UG_COEFF_INT * (u - U_OFFSET) - VG_COEFF_INT * (v - V_OFFSET) + CONVERT_DELTA)
                            >> CONVERT_FRACTION_BIT), 0, 255); // G value
            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2) = (uint8_t)
                    clamp(((YB_COEFF_INT * (y - Y_OFFSET) + UB_COEFF_INT * (u - U_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // B value
        }
    }
}

int RGB24_to_YV12_neon(YV12_BUFFER_CONFIG *ybf, RGB24_BUFFER_CONFIG *rbf, vpx_color_space_t color_space, vpx_color_range_t color_range) {
    if (ybf == NULL || rbf == NULL) {
        return -1;
    }

    if (color_space == VPX_CS_BT_709 && color_range == VPX_CR_STUDIO_RANGE) {
        RGB24_to_YV12_bt701_neon(ybf, rbf);
    }
    else {
        return -1;
    }

    return 0;
}

int YV12_to_RGB24_neon(RGB24_BUFFER_CONFIG *rbf, YV12_BUFFER_CONFIG *ybf, vpx_color_space_t color_space, vpx_color_range_t color_range) {
    if (ybf == NULL || rbf == NULL) {
        return -1;
    }

    if (color_space == VPX_CS_BT_709 && color_range == VPX_CR_STUDIO_RANGE) {
        YV12_to_RGB24_bt701_neon(rbf, ybf);
    }
    else {
        return -1;
    }

    return 0;
}