//
// Created by hyunho on 7/1/20.
//

#ifndef LIBVPX_WRAPPER_VPX_BILINEAR_H
#define LIBVPX_WRAPPER_VPX_BILINEAR_H

#define BILINEAR_FRACTION_BIT (5)
#define BILINEAR_FRACTION_SCALE (1 << BILINEAR_FRACTION_BIT)
static const int16_t BILINEAR_DELTA = (1 << (BILINEAR_FRACTION_BIT - 1));

#endif //LIBVPX_WRAPPER_VPX_BILINEAR_H
