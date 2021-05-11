#!/bin/sh
./vpxdec --codec=vp9 --summary --noblit --threads=1 --frame-buffers=50 --limit=30 \
    --content-dir=/home/hyunho/data-deprecated/movie \
    --input-video=270p_512k_60sec_125st.webm \
    --dnn-video=1080p_270p_60sec_125st_EDSR_transpose_B8_F32_S4.webm \
    --compare-video=1080p_lossless_60sec_125st.webm \
    --decode-mode=2 --dnn-mode=2 --cache-policy=2 --save-quality --save-frame
