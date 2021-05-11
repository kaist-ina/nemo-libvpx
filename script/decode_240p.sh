#!/bin/sh
../vpxdec --codec=vp9 --progress --summary --noblit --threads=1 --frame-buffers=50 --limit=120 \
    --content-dir=/ssd1/nemo-mobicom/game-lol \
    --input-video=240p_s0_d60_encoded.webm \
    --filter-interval=30 \
    --postfix=libvpx \
    --save-frame
