/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_dsp/skin_detection.h"
#include "vpx_util/vpx_write_yuv_frame.h"

int vpx_write_y_frame(char *file_path, YV12_BUFFER_CONFIG *s){
    FILE *y_file = fopen(file_path, "wb");
    if(y_file == NULL)
    {
        fprintf(stderr, "%s: fail to open a file %s", __func__, file_path);
        return -1;
    }

    unsigned char *src = s->y_buffer;
    int h = s->y_crop_height;

    do {
        fwrite(src, s->y_crop_width, 1, y_file);
        src += s->y_stride;
    } while (--h);

    fclose(y_file);
    return 0;
}


void vpx_write_yuv_frame(FILE *yuv_file, YV12_BUFFER_CONFIG *s) {
#if defined(OUTPUT_YUV_SRC) || defined(OUTPUT_YUV_DENOISED) || \
    defined(OUTPUT_YUV_SKINMAP)

  unsigned char *src = s->y_buffer;
  int h = s->y_crop_height;

  do {
    fwrite(src, s->y_width, 1, yuv_file);
    src += s->y_stride;
  } while (--h);

  src = s->u_buffer;
  h = s->uv_crop_height;

  do {
    fwrite(src, s->uv_width, 1, yuv_file);
    src += s->uv_stride;
  } while (--h);

  src = s->v_buffer;
  h = s->uv_crop_height;

  do {
    fwrite(src, s->uv_width, 1, yuv_file);
    src += s->uv_stride;
  } while (--h);

#else
  (void)yuv_file;
  (void)s;
#endif
}
