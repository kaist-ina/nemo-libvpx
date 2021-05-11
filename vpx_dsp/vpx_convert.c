//
// Created by hyunho on 6/30/20.
//

#include <stdint.h>
#include <math.h>
#include <time.h>
#include "vpx_dsp/vpx_convert.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_nemo.h"
#include "../vpx/vpx_image.h"
#include "../vpx_dsp/vpx_dsp_common.h"
#include "vpx_convert.h"
#include "../vpx_scale/yv12config.h"
#include "../vpx/vpx_nemo.h"

#define DEBUG_LATENCY 0
#define BILLION  1E9

//naive c implementation
//int RGB24_to_YV12_bt701_c(YV12_BUFFER_CONFIG *ybf, RGB24_BUFFER_CONFIG *rbf) {
//    uint8_t r, g, b;
//    int i, j;
//    const int height = ybf->y_crop_height;
//    const int width = ybf->y_crop_width;
//
//    for (i = 0; i < height; i++) {
//        for (j = 0; j < width; j++) {
//            r = *(rbf->buffer_alloc + i * rbf->stride + j * 3);
//            g = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1);
//            b = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2);
//
//            *(ybf->y_buffer + i * ybf->y_stride + j) = (uint8_t) clamp(round((RY_COEFF_FLOAT * r + GY_COEFF_FLOAT * g + BY_COEFF_FLOAT * b) + Y_OFFSET), 0, 255);
//            *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x)) = (uint8_t) clamp(
//                    round((-RU_COEFF_FLOAT * r - GU_COEFF_FLOAT * g + BU_COEFF_FLOAT * b) + U_OFFSET), 0, 255);
//            *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x)) = (uint8_t) clamp(
//                    round((RV_COEFF_FLOAT * r - GV_COEFF_FLOAT * g - BV_COEFF_FLOAT * b) + V_OFFSET), 0, 255);
//        }
//    }
//
//    return 0;
//}

//naive c implementation
//int YV12_to_RGB24_bt701_c(YV12_BUFFER_CONFIG *ybf, RGB24_BUFFER_CONFIG *rbf) {
//    uint8_t y, u, v;
//    int i, j;
//    const int height = ybf->y_crop_height;
//    const int width = ybf->y_crop_width;
//
//    for (i = 0; i < height; i++) {
//        for (j = 0; j < width; j++) {
//            y = *(ybf->y_buffer + i * ybf->y_stride + j);
//            u = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));
//            v = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));
//
//            *(rbf->buffer_alloc + i * rbf->stride + j * 3) = (uint8_t) clamp(round(YR_COEFF_FLOAT * (y - Y_OFFSET) + VR_COEFF_FLOAT * (v - V_OFFSET)), 0,
//                                                                             255); // R value
//            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1) = (uint8_t) clamp(
//                    round(YG_COEFF_FLOAT * (y - Y_OFFSET) - UG_COEFF_FLOAT * (u - U_OFFSET) - VG_COEFF_FLOAT * (v - V_OFFSET)), 0, 255); // G value
//            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2) = (uint8_t) clamp(round(YB_COEFF_FLOAT * (y - Y_OFFSET) + UB_COEFF_FLOAT * (u - U_OFFSET)), 0,
//                                                                                 255); // B value
//        }
//    }
//
//    return 0;
//}

int RGB24_float_to_uint8_c(RGB24_BUFFER_CONFIG *rbf) {
    if (rbf == NULL) {
        return -1;
    }

    float *src = rbf->buffer_alloc_float;
    uint8_t *dst = rbf->buffer_alloc;

    int w, h;
    for (h = 0; h < rbf->height; h++) {
        for (w = 0; w < rbf->width; w++) {
            *(dst + w + h * rbf->stride) = clamp(round(*(src + w + h * rbf->stride)), 0, 255);
        }
    }

    return 0;
}

//optimization: fixed-point operation
void RGB24_to_YV12_bt701_c(YV12_BUFFER_CONFIG *ybf, RGB24_BUFFER_CONFIG *rbf) {
    uint8_t r, g, b;
    uint8_t r1, g1, b1;
    uint8_t r2, g2, b2;
    uint8_t r3, g3, b3;

    int i, j;
    const int height = ybf->y_crop_height;
    const int width = ybf->y_crop_width;
    for (i = 0; i < height; i++) {
        for (j = 0; j <= width - 4 ; j += 4) {
            r = *(rbf->buffer_alloc + i * rbf->stride + j * 3);
            g = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1);
            b = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2);
            r1 = *(rbf->buffer_alloc + i * rbf->stride + (j + 1) * 3);
            g1 = *(rbf->buffer_alloc + i * rbf->stride + (j + 1) * 3 + 1);
            b1 = *(rbf->buffer_alloc + i * rbf->stride + (j + 1) * 3 + 2);
            r2 = *(rbf->buffer_alloc + i * rbf->stride + (j + 2) * 3);
            g2 = *(rbf->buffer_alloc + i * rbf->stride + (j + 2) * 3 + 1);
            b2 = *(rbf->buffer_alloc + i * rbf->stride + (j + 2) * 3 + 2);
            r3 = *(rbf->buffer_alloc + i * rbf->stride + (j + 3) * 3);
            g3 = *(rbf->buffer_alloc + i * rbf->stride + (j + 3) * 3 + 1);
            b3 = *(rbf->buffer_alloc + i * rbf->stride + (j + 3) * 3 + 2);

            *(ybf->y_buffer + i * ybf->y_stride + j) = (uint8_t) (
                    ((RY_COEFF_INT * r + GY_COEFF_INT * g + BY_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + Y_OFFSET);
            *(ybf->y_buffer + i * ybf->y_stride + j + 1) = (uint8_t) (
                    ((RY_COEFF_INT * r1 + GY_COEFF_INT * g1 + BY_COEFF_INT * b1 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) +
                    Y_OFFSET);
            *(ybf->y_buffer + i * ybf->y_stride + j + 2) = (uint8_t) (
                    ((RY_COEFF_INT * r2 + GY_COEFF_INT * g2 + BY_COEFF_INT * b2 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) +
                    Y_OFFSET);
            *(ybf->y_buffer + i * ybf->y_stride + j + 3) = (uint8_t) (
                    ((RY_COEFF_INT * r3 + GY_COEFF_INT * g3 + BY_COEFF_INT * b3 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) +
                    Y_OFFSET);
        }

        for (; j < width; j++) {
            r = *(rbf->buffer_alloc + i * rbf->stride + j * 3);
            g = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1);
            b = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2);
            *(ybf->y_buffer + i * ybf->y_stride + j) = (uint8_t) (
                    ((RY_COEFF_INT * r + GY_COEFF_INT * g + BY_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + Y_OFFSET);
        }
    }

    for (i = 0; i < height; i += (1 << ybf->subsampling_y)) {
        for (j = 0; j <= width - (4 << ybf->subsampling_x); j += (4 << ybf->subsampling_x)) {
            r = *(rbf->buffer_alloc + i * rbf->stride + j * 3);
            g = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1);
            b = *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2);
            r1 = *(rbf->buffer_alloc + i * rbf->stride + (j + (1 << ybf->subsampling_x)) * 3);
            g1 = *(rbf->buffer_alloc + i * rbf->stride + (j + (1 << ybf->subsampling_x)) * 3 + 1);
            b1 = *(rbf->buffer_alloc + i * rbf->stride + (j + (1 << ybf->subsampling_x)) * 3 + 2);
            r2 = *(rbf->buffer_alloc + i * rbf->stride + (j + (2 << ybf->subsampling_x)) * 3);
            g2 = *(rbf->buffer_alloc + i * rbf->stride + (j + (2 << ybf->subsampling_x)) * 3 + 1);
            b2 = *(rbf->buffer_alloc + i * rbf->stride + (j + (2 << ybf->subsampling_x)) * 3 + 2);
            r3 = *(rbf->buffer_alloc + i * rbf->stride + (j + (3 << ybf->subsampling_x)) * 3);
            g3 = *(rbf->buffer_alloc + i * rbf->stride + (j + (3 << ybf->subsampling_x)) * 3 + 1);
            b3 = *(rbf->buffer_alloc + i * rbf->stride + (j + (3 << ybf->subsampling_x)) * 3 + 2);

            *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x)) = (uint8_t) (
                    ((-RU_COEFF_INT * r - GU_COEFF_INT * g + BU_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + U_OFFSET);
            *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 1) = (uint8_t) (
                    ((-RU_COEFF_INT * r1 - GU_COEFF_INT * g1 + BU_COEFF_INT * b1 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + U_OFFSET);
            *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 2) = (uint8_t) (
                    ((-RU_COEFF_INT * r2 - GU_COEFF_INT * g2 + BU_COEFF_INT * b2 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + U_OFFSET);
            *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 3) = (uint8_t) (
                    ((-RU_COEFF_INT * r3 - GU_COEFF_INT * g3 + BU_COEFF_INT * b3 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + U_OFFSET);
            *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x)) = (uint8_t) (
                    ((RV_COEFF_INT * r - GV_COEFF_INT * g - BV_COEFF_INT * b + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + V_OFFSET);
            *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 1) = (uint8_t) (
                    ((RV_COEFF_INT * r1 - GV_COEFF_INT * g1 - BV_COEFF_INT * b1 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + V_OFFSET);
            *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 2) = (uint8_t) (
                    ((RV_COEFF_INT * r2 - GV_COEFF_INT * g2 - BV_COEFF_INT * b2 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + V_OFFSET);
            *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x) + 3) = (uint8_t) (
                    ((RV_COEFF_INT * r3 - GV_COEFF_INT * g3 - BV_COEFF_INT * b3 + CONVERT_DELTA) >> CONVERT_FRACTION_BIT) + V_OFFSET);
        }


        for (; j < width; j += (1 << ybf->subsampling_x)) {
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

void YV12_to_RGB24_bt701_c(RGB24_BUFFER_CONFIG *rbf, YV12_BUFFER_CONFIG *ybf) {
    uint8_t y, u, v;
    uint8_t y1, u1, v1;
    uint8_t y2, u2, v2;
    uint8_t y3, u3, v3;

    int i, j;
    const int height = ybf->y_crop_height;
    const int width = ybf->y_crop_width;
    for (i = 0; i < height; i++) {
        for (j = 0; j <= width - 4 ; j += 4) {
            y = *(ybf->y_buffer + i * ybf->y_stride + j);
            u = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));
            v = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));
            y1 = *(ybf->y_buffer + i * ybf->y_stride + j + 1);
            u1 = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 1) >> ybf->subsampling_x));
            v1 = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 1) >> ybf->subsampling_x));
            y2 = *(ybf->y_buffer + i * ybf->y_stride + j + 2);
            u2 = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 2) >> ybf->subsampling_x));
            v2 = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 2) >> ybf->subsampling_x));
            y3 = *(ybf->y_buffer + i * ybf->y_stride + j + 3);
            u3 = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 3) >> ybf->subsampling_x));
            v3 = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + ((j + 3) >> ybf->subsampling_x));

            *(rbf->buffer_alloc + i * rbf->stride + j * 3) =
                    clamp(((YR_COEFF_INT * (y - Y_OFFSET) + VR_COEFF_INT * (v - V_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // R value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 1) * 3) =
                    clamp(((YR_COEFF_INT * (y1 - Y_OFFSET) + VR_COEFF_INT * (v1 - V_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // R value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 2) * 3) =
                    clamp(((YR_COEFF_INT * (y2 - Y_OFFSET) + VR_COEFF_INT * (v2 - V_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // R value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 3) * 3) =
                    clamp(((YR_COEFF_INT * (y3 - Y_OFFSET) + VR_COEFF_INT * (v3 - V_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // R value

            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1) =
                    clamp(((YG_COEFF_INT * (y - Y_OFFSET) - UG_COEFF_INT * (u - U_OFFSET) - VG_COEFF_INT * (v - V_OFFSET) + CONVERT_DELTA)
                            >> CONVERT_FRACTION_BIT), 0, 255); // G value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 1) * 3 + 1) =
                    clamp(((YG_COEFF_INT * (y1 - Y_OFFSET) - UG_COEFF_INT * (u1 - U_OFFSET) - VG_COEFF_INT * (v1 - V_OFFSET) + CONVERT_DELTA)
                            >> CONVERT_FRACTION_BIT), 0, 255); // G value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 2) * 3 + 1) =
                    clamp(((YG_COEFF_INT * (y2 - Y_OFFSET) - UG_COEFF_INT * (u2 - U_OFFSET) - VG_COEFF_INT * (v2 - V_OFFSET) + CONVERT_DELTA)
                            >> CONVERT_FRACTION_BIT), 0, 255); // G value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 3) * 3 + 1) =
                    clamp(((YG_COEFF_INT * (y3 - Y_OFFSET) - UG_COEFF_INT * (u3 - U_OFFSET) - VG_COEFF_INT * (v3 - V_OFFSET) + CONVERT_DELTA)
                            >> CONVERT_FRACTION_BIT), 0, 255); // G value

            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2) =
                    clamp(((YB_COEFF_INT * (y - Y_OFFSET) + UB_COEFF_INT * (u - U_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // B value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 1) * 3 + 2) =
                    clamp(((YB_COEFF_INT * (y1 - Y_OFFSET) + UB_COEFF_INT * (u1 - U_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // B value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 2) * 3 + 2) =
                    clamp(((YB_COEFF_INT * (y2 - Y_OFFSET) + UB_COEFF_INT * (u2 - U_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // B value
            *(rbf->buffer_alloc + i * rbf->stride + (j + 3) * 3 + 2) =
                    clamp(((YB_COEFF_INT * (y3 - Y_OFFSET) + UB_COEFF_INT * (u3 - U_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // B value
        }

        for (; j < width; j++) {
            y = *(ybf->y_buffer + i * ybf->y_stride + j);
            u = *(ybf->u_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));
            v = *(ybf->v_buffer + (i >> ybf->subsampling_y) * ybf->uv_stride + (j >> ybf->subsampling_x));

            *(rbf->buffer_alloc + i * rbf->stride + j * 3) =
                    clamp(((YR_COEFF_INT * (y - Y_OFFSET) + VR_COEFF_INT * (v - V_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // R value
            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 1) =
                    clamp(((YG_COEFF_INT * (y - Y_OFFSET) - UG_COEFF_INT * (u - U_OFFSET) - VG_COEFF_INT * (v - V_OFFSET) + CONVERT_DELTA)
                            >> CONVERT_FRACTION_BIT), 0, 255); // G value
            *(rbf->buffer_alloc + i * rbf->stride + j * 3 + 2) =
                    clamp(((YB_COEFF_INT * (y - Y_OFFSET) + UB_COEFF_INT * (u - U_OFFSET) + CONVERT_DELTA) >> CONVERT_FRACTION_BIT), 0, 255); // B value
        }
    }
}


int RGB24_to_YV12_c(YV12_BUFFER_CONFIG *ybf, RGB24_BUFFER_CONFIG *rbf, vpx_color_space_t color_space, vpx_color_range_t color_range) {
    if (ybf == NULL || rbf == NULL) {
        return -1;
    }

#if DEBUG_LATENCY
    struct timespec start_time, finish_time;
    double diff;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
#endif
    if (color_space == VPX_CS_BT_709 && color_range == VPX_CR_STUDIO_RANGE) {
        RGB24_to_YV12_bt701_c(ybf, rbf);
    }
    else {
        return -1;
    }
#if DEBUG_LATENCY
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    diff = (finish_time.tv_sec - start_time.tv_sec) * 1000 +
           (finish_time.tv_nsec - start_time.tv_nsec) / BILLION * 1000.0;
    printf("rgb24_to_yv12: %f", diff);
#endif

    return 0;
}

int YV12_to_RGB24_c(RGB24_BUFFER_CONFIG *rbf, YV12_BUFFER_CONFIG *ybf, vpx_color_space_t color_space, vpx_color_range_t color_range) {
    if (ybf == NULL || rbf == NULL) {
        return -1;
    }

#if DEBUG_LATENCY
    struct timespec start_time, finish_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    double diff;
#endif
    if (color_space == VPX_CS_BT_709 && color_range == VPX_CR_STUDIO_RANGE) {
        YV12_to_RGB24_bt701_c(rbf, ybf);
    }
    else {
        return -1;
    }
#if DEBUG_LATENCY
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    diff = (finish_time.tv_sec - start_time.tv_sec) * 1000 +
           (finish_time.tv_nsec - start_time.tv_nsec) / BILLION * 1000.0;
    printf("yv12_to_rgb24: %f", diff);
#endif

    return 0;
}
