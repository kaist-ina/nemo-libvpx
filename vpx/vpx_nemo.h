//
// Created by hyunho on 9/2/19.
//

#ifndef LIBVPX_WRAPPER_VPX_SR_CACHE_H
#define LIBVPX_WRAPPER_VPX_SR_CACHE_H

#include <limits.h>
#include "./vpx_config.h"
#include "vpx_scale/yv12config.h"

/* Structure to log latency */
typedef struct nemo_latency {
    //decode
    double decode;
    double decode_intra_block;
    double decode_inter_block;
    double decode_inter_residual;

    //interpolation (interp)
    double interp;
    double interp_intra_block;
    double interp_inter_block;
    double interp_inter_residual;

    //super-resolution (sr)
    double sr;
    double sr_convert_rgb_to_yuv;
    double sr_execute_dnn;
    double sr_convert_yuv_to_rgb;
    double sr_convert_float_to_int;
} nemo_latency_t;

/* Structure to log frame index */
typedef struct nemo_frame {
    int video_frame_index;
    int super_frame_index;
} nemo_frame_t;

/* Structure to log metdata */
typedef struct nemo_metadata {
    nemo_frame_t reference_frames[3];
    int num_blocks;
    int num_intrablocks;
    int num_interblocks;
    int num_noskip_interblocks;
} nemo_metdata_t;

/* Enum to set a decode mode */
typedef enum{
    DECODE,
    DECODE_SR,
    DECODE_CACHE,
} nemo_decode_mode;

/* Enum to set a cache mode */
typedef enum{
    NO_CACHE,
    PROFILE_CACHE,
    KEY_FRAME_CACHE,
} nemo_cache_mode;

/* Enum to set a dnn mode */
typedef enum{
    NO_DNN,
    ONLINE_DNN,
    OFFLINE_DNN,
} nemo_dnn_mode;

/* Enum to set a DNN runtime */
typedef enum{
    CPU_FLOAT32,
    GPU_FLOAT32_16_HYBRID,
    DSP_FIXED8,
    GPU_FLOAT16,
    AIP_FIXED8
} nemo_dnn_runtime;

/* Structure to hold coefficients to run bilinear interpolation */
typedef struct nemo_bilinear_coeff{
    float *x_lerp;
    int16_t *x_lerp_fixed;
    float *y_lerp;
    int16_t *y_lerp_fixed;
    int *top_y_index;
    int *bottom_y_index;
    int *left_x_index;
    int *right_x_index;
} nemo_bilinear_coeff_t;

/* Structure to read a cache profile */
//TODO: support file_size
typedef struct nemo_cache_profile {
    FILE *file;
    uint64_t offset;
    uint8_t byte_value;
    off_t file_size;
    int num_dummy_bits;
} nemo_cache_profile_t;

/* Structure to hold the location of block, which will be interpolated */
typedef struct nemo_interp_block{
    int mi_row;
    int mi_col;
    int n4_w[3];
    int n4_h[3];
    struct nemo_interp_block *next;
} nemo_interp_block_t;

/* Structure to hold a list of block locations */
typedef struct nemo_interp_block_list{
    nemo_interp_block_t *cur;
    nemo_interp_block_t *head;
    nemo_interp_block_t *tail;
} nemo_interp_block_list_t;

/* Structure to hold per-thread information  */
typedef struct nemo_worker_data {
    //interpolation
    YV12_BUFFER_CONFIG *lr_resiudal;
    nemo_interp_block_list_t *intra_block_list;
    nemo_interp_block_list_t *inter_block_list;

    //log
    int index;
    FILE *latency_log;
    FILE *metadata_log;
    nemo_latency_t latency;
    nemo_metdata_t metadata;
} nemo_worker_data_t;

/* Structure for a RGB888 frame  */
typedef struct rgb24_buffer_config{
    int width;
    int height;
    int stride;
    int buffer_alloc_sz;
    uint8_t *buffer_alloc;
    float *buffer_alloc_float; //TODO: remove this
} RGB24_BUFFER_CONFIG;

typedef struct nemo_dnn{
    void *interpreter;
    int scale;
} nemo_dnn_t;

/* Struture to hold per-decoder information */
typedef struct nemo_cfg{
    //direcetory
    char log_dir[PATH_MAX];
    char input_frame_dir[PATH_MAX];
    char sr_frame_dir[PATH_MAX];
    char input_reference_frame_dir[PATH_MAX];
    char sr_reference_frame_dir[PATH_MAX];
    char sr_offline_frame_dir[PATH_MAX]; //OFFLINE_DNN (load images)

    //logging
    int save_rgbframe; // rgb
    int save_yuvframe;
    int save_quality;
    int save_latency;
    int save_metadata;
    int filter_interval;

    //mode
    nemo_decode_mode decode_mode;
    nemo_cache_mode cache_mode;
    nemo_dnn_mode dnn_mode;
    nemo_dnn_runtime dnn_runtime;
    int target_width;
    int target_height;

    //profile
    nemo_dnn_t *dnn;
    nemo_cache_profile_t *cache_profile;
    nemo_bilinear_coeff_t *bilinear_coeff;
} nemo_cfg_t;


#ifdef __cplusplus
extern "C" {
#endif

nemo_cfg_t *init_nemo_cfg();
void remove_nemo_cfg(nemo_cfg_t *config);

nemo_dnn_t *init_nemo_dnn(int scale);
void remove_nemo_dnn(nemo_dnn_t *dnn);

nemo_cache_profile_t *init_nemo_cache_profile();
void remove_nemo_cache_profile(nemo_cache_profile_t *cache_profile);
int read_cache_profile(nemo_cache_profile_t *profile);
int read_cache_profile_dummy_bits(nemo_cache_profile_t *cache_profile);

nemo_bilinear_coeff_t *init_bilinear_coeff(int width, int height, int scale);
void remove_bilinear_coeff(nemo_bilinear_coeff_t *coeff);

void create_nemo_interp_block(struct nemo_interp_block_list *L, int mi_col, int mi_row, int n4_w,
                              int n4_h);
void set_nemo_interp_block(struct nemo_interp_block_list *L, int plane, int n4_w, int n4_h);

nemo_worker_data_t *init_nemo_worker(int num_threads, nemo_cfg_t *nemo_cfg);
void remove_nemo_worker(nemo_worker_data_t *mwd, int num_threads);

int RGB24_save_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path);
int RGB24_load_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path);
int RGB24_alloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height);
int RGB24_realloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height);
int RGB24_free_frame_buffer(RGB24_BUFFER_CONFIG *rbf);
double RGB24_calc_psnr(const RGB24_BUFFER_CONFIG *a, const RGB24_BUFFER_CONFIG *b);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //LIBVPX_WRAPPER_VPX_SR_CACHE_H
