//
// Created by hyunho on 9/2/19.
//
#include <time.h>
#include <memory.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include <vpx_dsp/psnr.h>
#include <vpx_dsp/vpx_dsp_common.h>
#include <vpx_scale/yv12config.h>
#include <vpx_mem/vpx_mem.h>
#include <sys/param.h>
#include <math.h>
#include <libgen.h>

#include "third_party/libyuv/include/libyuv/convert.h"
#include "third_party/libyuv/include/libyuv/convert_from.h"
//#include "third_party/libyuv/include/libyuv/scale.h"

#include "vpx/vpx_nemo.h"

#ifdef __ANDROID_API__

#include <android/log.h>
#include <arm_neon.h>

#define TAG "LoadInputTensor JNI"
#define _UNKNOWN   0
#define _DEFAULT   1
#define _VERBOSE   2
#define _DEBUG    3
#define _INFO        4
#define _WARN        5
#define _ERROR    6
#define _FATAL    7
#define _SILENT       8
#define LOGUNK(...) __android_log_print(_UNKNOWN,TAG,__VA_ARGS__)
#define LOGDEF(...) __android_log_print(_DEFAULT,TAG,__VA_ARGS__)
#define LOGV(...) __android_log_print(_VERBOSE,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(_DEBUG,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(_INFO,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(_WARN,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(_ERROR,TAG,__VA_ARGS__)
#define LOGF(...) __android_log_print(_FATAL,TAG,__VA_ARGS__)
#define LOGS(...) __android_log_print(_SILENT,TAG,__VA_ARGS__)
#endif


nemo_cfg_t *init_nemo_cfg() {
    nemo_cfg_t *config = (nemo_cfg_t *) vpx_calloc(1, sizeof(nemo_cfg_t));
    return config;
}

void remove_nemo_cfg(nemo_cfg_t *config) {
    if (config) {
        remove_nemo_dnn(config->dnn);
        remove_nemo_cache_profile(config->cache_profile);
        remove_bilinear_coeff(config->bilinear_coeff);
        vpx_free(config);
    }
}

nemo_dnn_t *init_nemo_dnn(int scale) {
    nemo_dnn_t *profile = (nemo_dnn_t *) vpx_calloc(1, sizeof(nemo_dnn_t));
    profile->interpreter = NULL;
    profile->scale = scale;

    return profile;
}

void remove_nemo_dnn(nemo_dnn_t *dnn) {
    if (dnn) {
        if (dnn->interpreter) {
#if CONFIG_SNPE
            snpe_free(dnn->interpreter);
#endif
        }
        vpx_free(dnn);
    }
}

nemo_cache_profile_t *init_nemo_cache_profile() {
    nemo_cache_profile_t *profile = (nemo_cache_profile_t *) vpx_calloc(1, sizeof(nemo_cache_profile_t));
    profile->file = NULL;
    profile->num_dummy_bits = 0;

    return profile;
}

void remove_nemo_cache_profile(nemo_cache_profile_t *cache_profile) {
    if (cache_profile) {
        if (cache_profile->file) fclose(cache_profile->file);
        vpx_free(cache_profile);
    }
}

int read_cache_profile_dummy_bits(nemo_cache_profile_t *cache_profile) {
    int i, dummy;

    if (cache_profile == NULL) {
        fprintf(stderr, "%s: cache_profile is NULL", __func__);
        return -1;
    }
    if (cache_profile->file == NULL) {
        fprintf(stderr, "%s: cache_profile->file is NULL", __func__);
        return -1;
    }

    if (cache_profile->num_dummy_bits > 0) {
        for (i = 0; i < cache_profile->num_dummy_bits; i++) {
            cache_profile->offset += 1;
        }
    }

    if (fread(&cache_profile->num_dummy_bits, sizeof(int), 1, cache_profile->file) != 1) {
        fprintf(stderr, "%s: fail to read a cache profile\n", __func__);
        return -1;
    }

    return 0;
}

int read_cache_profile(nemo_cache_profile_t *profile) {
    uint8_t apply_dnn;


    if (profile == NULL) {
        fprintf(stderr, "%s: profile is NULL", __func__);
        return -1;
    }
    if (profile->file == NULL) {
        fprintf(stderr, "%s: profile->file is NULL", __func__);
        return -1;
    }

    if (profile->offset % 8 == 0) {
        if (fread(&profile->byte_value, sizeof(uint8_t), 1, profile->file) != 1) {
            fprintf(stderr, "%s: fail to read a cache profile\n", __func__);
            return -1;
        }
    }

    apply_dnn = (profile->byte_value & (1 << (profile->offset % 8))) >> (profile->offset % 8); //TODO: 1, 0
    profile->offset += 1;

    return apply_dnn;
}

void remove_nemo_worker(nemo_worker_data_t *mwd, int num_threads) {
    int i;
    if (mwd != NULL) {
        for (i = 0; i < num_threads; ++i) {
            vpx_free_frame_buffer(mwd[i].lr_resiudal);
            vpx_free(mwd[i].lr_resiudal);

            //free decode block lists
            nemo_interp_block_t *intra_block = mwd[i].intra_block_list->head;
            nemo_interp_block_t *prev_block = NULL;
            while (intra_block != NULL) {
                prev_block = intra_block;
                intra_block = intra_block->next;
                vpx_free(prev_block);
            }
            vpx_free(mwd[i].intra_block_list);

            nemo_interp_block_t *inter_block = mwd[i].inter_block_list->head;
            while (inter_block != NULL) {
                prev_block = inter_block;
                inter_block = inter_block->next;
                vpx_free(prev_block);
            }
            vpx_free(mwd[i].inter_block_list);

            if (mwd[i].latency_log != NULL) fclose(mwd[i].latency_log);
            if (mwd[i].metadata_log != NULL) fclose(mwd[i].metadata_log);
        }
        vpx_free(mwd);
    }
}

static void init_nemo_worker_data(nemo_worker_data_t *mwd, int index) {
    assert (mwd != NULL);

    mwd->lr_resiudal = (YV12_BUFFER_CONFIG *) vpx_calloc(1, sizeof(YV12_BUFFER_CONFIG));

    mwd->intra_block_list = (nemo_interp_block_list_t *) vpx_calloc(1, sizeof(nemo_interp_block_list_t));
    mwd->intra_block_list->cur = NULL;
    mwd->intra_block_list->head = NULL;
    mwd->intra_block_list->tail = NULL;

    mwd->inter_block_list = (nemo_interp_block_list_t *) vpx_calloc(1, sizeof(nemo_interp_block_list_t));
    mwd->inter_block_list->cur = NULL;
    mwd->inter_block_list->head = NULL;
    mwd->inter_block_list->tail = NULL;

    mwd->index = index;

    mwd->latency_log = NULL;
    mwd->metadata_log = NULL;
}

nemo_worker_data_t *init_nemo_worker(int num_threads, nemo_cfg_t *nemo_cfg) {
    char latency_log_path[PATH_MAX];
    char metadata_log_path[PATH_MAX];

    if (!nemo_cfg) {
        fprintf(stderr, "%s: nemo_cfg is NULL", __func__);
        return NULL;
    }
    if (num_threads <= 0) {
        fprintf(stderr, "%s: num_threads is equal or less than 0", __func__);
        return NULL;
    }

    nemo_worker_data_t *mwd = (nemo_worker_data_t *) vpx_malloc(sizeof(nemo_worker_data_t) * num_threads);
    int i;
    for (i = 0; i < num_threads; ++i) {
        init_nemo_worker_data(&mwd[i], i);

        if (nemo_cfg->save_latency == 1) {
            sprintf(latency_log_path, "%s/latency_thread%d%d.txt", nemo_cfg->log_dir, mwd[i].index, num_threads);
            if ((mwd[i].latency_log = fopen(latency_log_path, "w")) == NULL) {
                fprintf(stderr, "%s: cannot open a file %s", __func__, latency_log_path);
                nemo_cfg->save_latency = 0;
            }
        }

        if (nemo_cfg->save_metadata == 1) {
            sprintf(metadata_log_path, "%s/metadata_thread%d%d.txt", nemo_cfg->log_dir, mwd[i].index, num_threads);
            if ((mwd[i].metadata_log = fopen(metadata_log_path, "w")) == NULL) {
                fprintf(stderr, "%s: cannot open a file %s", __func__, metadata_log_path);
                nemo_cfg->save_metadata = 0;
            }
        }
    }

    return mwd;
}


nemo_bilinear_coeff_t *init_bilinear_coeff(int width, int height, int scale) {
    struct nemo_bilinear_coeff *coeff = (nemo_bilinear_coeff_t *) vpx_calloc(1, sizeof(nemo_bilinear_coeff_t));
    int x, y;

    assert (coeff != NULL);
    assert (width != 0 && height != 0 && scale > 0);

    coeff->x_lerp = (float *) vpx_malloc(sizeof(float) * width * scale);
    coeff->x_lerp_fixed = (int16_t *) vpx_malloc(sizeof(int16_t) * width * scale);
    coeff->left_x_index = (int *) vpx_malloc(sizeof(int) * width * scale);
    coeff->right_x_index = (int *) vpx_malloc(sizeof(int) * width * scale);

    coeff->y_lerp = (float *) vpx_malloc(sizeof(float) * height * scale);
    coeff->y_lerp_fixed = (int16_t *) vpx_malloc(sizeof(int16_t) * height * scale);
    coeff->top_y_index = (int *) vpx_malloc(sizeof(int) * height * scale);
    coeff->bottom_y_index = (int *) vpx_malloc(sizeof(int) * height * scale);

    for (x = 0; x < width * scale; ++x) {
        const double in_x = (x + 0.5f) / scale - 0.5f;
        coeff->left_x_index[x] = MAX(floor(in_x), 0);
        coeff->right_x_index[x] = MIN(ceil(in_x), width - 1);
        coeff->x_lerp[x] = in_x - floor(in_x);
        coeff->x_lerp_fixed[x] = coeff->x_lerp[x] * 32;
    }

    for (y = 0; y < height * scale; ++y) {
        const double in_y = (y + 0.5f) / scale - 0.5f;
        coeff->top_y_index[y] = MAX(floor(in_y), 0);
        coeff->bottom_y_index[y] = MIN(ceil(in_y), height - 1);
        coeff->y_lerp[y] = in_y - floor(in_y);
        coeff->y_lerp_fixed[y] = coeff->y_lerp[y] * 32;
    }

    return coeff;
}

void remove_bilinear_coeff(nemo_bilinear_coeff_t *coeff) {
    if (coeff != NULL) {
        vpx_free(coeff->x_lerp);
        vpx_free(coeff->x_lerp_fixed);
        vpx_free(coeff->left_x_index);
        vpx_free(coeff->right_x_index);

        vpx_free(coeff->y_lerp);
        vpx_free(coeff->y_lerp_fixed);
        vpx_free(coeff->top_y_index);
        vpx_free(coeff->bottom_y_index);

        vpx_free(coeff);
    }
}

void create_nemo_interp_block(nemo_interp_block_list_t *L, int mi_col, int mi_row, int n4_w,
                              int n4_h) {
    nemo_interp_block_t *newBlock = (nemo_interp_block_t *) vpx_calloc(1, sizeof(nemo_interp_block_t));
    newBlock->mi_col = mi_col;
    newBlock->mi_row = mi_row;
    newBlock->n4_w[0] = n4_w;
    newBlock->n4_h[0] = n4_h;
    newBlock->next = NULL;

    if (L->head == NULL && L->tail == NULL) {
        L->head = L->tail = newBlock;
    } else {
        L->tail->next = newBlock;
        L->tail = newBlock;
    }

    L->cur = newBlock;
}

void set_nemo_interp_block(nemo_interp_block_list_t *L, int plane, int n4_w, int n4_h) {
    nemo_interp_block_t *currentBlock = L->cur;
    currentBlock->n4_w[plane] = n4_w;
    currentBlock->n4_h[plane] = n4_h;
}

int RGB24_save_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path) {
    FILE *serialize_file = fopen(file_path, "wb");
    if (serialize_file == NULL) {
        fprintf(stderr, "%s: fail to save a file to %s\n", __func__, file_path);
        return -1;
    }

    uint8_t *src = rbf->buffer_alloc;
    int h = rbf->height;
    do {
        fwrite(src, sizeof(uint8_t), rbf->width, serialize_file);
        src += rbf->stride;
    } while (--h);

    fclose(serialize_file);

    return 0;
}

int RGB24_load_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path) {
    FILE *serialize_file = fopen(file_path, "rb");
    if (serialize_file == NULL) {
        fprintf(stderr, "%s: fail to open a file from %s\n", __func__, file_path);
        return -1;
    }

    uint8_t *src = rbf->buffer_alloc;
    int h = rbf->height;
    do {
        fread(src, sizeof(uint8_t), rbf->width, serialize_file);
        src += rbf->stride;
    } while (--h);

    fclose(serialize_file);

    return 0;
}

int RGB24_alloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height) {
    if (rbf) {
        RGB24_free_frame_buffer(rbf);
        return RGB24_realloc_frame_buffer(rbf, width, height);
    }
    return -1;
}

int RGB24_realloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height) {
    if (rbf) {
        const int stride = width * 3;

        const int frame_size = height * stride;

        if (frame_size > rbf->buffer_alloc_sz) {
            if (rbf->buffer_alloc_sz != 0) {
                vpx_free(rbf->buffer_alloc);
                vpx_free(rbf->buffer_alloc_float);
            }

            rbf->buffer_alloc = (uint8_t *) vpx_calloc(1, (size_t) frame_size * sizeof(uint8_t));
            if (!rbf->buffer_alloc) {
                return -1;
            }

            rbf->buffer_alloc_float = (float *) vpx_calloc(1, (size_t) frame_size * sizeof(float));
            if (!rbf->buffer_alloc_float) {
                return -1;
            }

            rbf->buffer_alloc_sz = (int) frame_size;
        }
        rbf->height = height;
        rbf->width = width * 3;
        rbf->stride = stride;

        return 0;
    }
    return -1;
}

int RGB24_free_frame_buffer(RGB24_BUFFER_CONFIG *rbf) {
    if (rbf) {
        if (rbf->buffer_alloc_sz > 0) {
            vpx_free(rbf->buffer_alloc);
            vpx_free(rbf->buffer_alloc_float);
        }
        memset(rbf, 0, sizeof(RGB24_BUFFER_CONFIG));
    } else {
        return -1;
    }
    return 0;
}

//from <vpx_dsp/src/psnr.c>
static void encoder_variance(const uint8_t *a, int a_stride, const uint8_t *b,
                             int b_stride, int w, int h, unsigned int *sse,
                             int *sum) {
    int i, j;

    *sum = 0;
    *sse = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            const int diff = a[j] - b[j];
            *sum += diff;
            *sse += diff * diff;
        }

        a += a_stride;
        b += b_stride;
    }
}

//from <vpx_dsp/src/psnr.c>
static int64_t get_sse(const uint8_t *a, int a_stride, const uint8_t *b,
                       int b_stride, int width, int height) {
    const int dw = width % 16;
    const int dh = height % 16;
    int64_t total_sse = 0;
    unsigned int sse = 0;
    int sum = 0;
    int x, y;

    if (dw > 0) {
        encoder_variance(&a[width - dw], a_stride, &b[width - dw], b_stride, dw,
                         height, &sse, &sum);
        total_sse += sse;
    }

    if (dh > 0) {
        encoder_variance(&a[(height - dh) * a_stride], a_stride,
                         &b[(height - dh) * b_stride], b_stride, width - dw, dh,
                         &sse, &sum);
        total_sse += sse;
    }

    for (y = 0; y < height / 16; ++y) {
        const uint8_t *pa = a;
        const uint8_t *pb = b;
        for (x = 0; x < width / 16; ++x) {
            vpx_mse16x16(pa, a_stride, pb, b_stride, &sse);
            total_sse += sse;

            pa += 16;
            pb += 16;
        }

        a += 16 * a_stride;
        b += 16 * b_stride;
    }

    return total_sse;
}

double RGB24_calc_psnr(const RGB24_BUFFER_CONFIG *a, const RGB24_BUFFER_CONFIG *b) {
    static const double peak = 255.0;
    double psnr;

    const int w = a->width;
    const int h = a->height;
    const uint32_t samples = w * h;
    const uint64_t sse = get_sse(a->buffer_alloc, a->stride, b->buffer_alloc, b->stride, w, h);

    psnr = vpx_sse_to_psnr(samples, peak, (double) sse);

    return psnr;
}
