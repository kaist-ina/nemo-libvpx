#!/bin/sh
./configure --enable-vp8 --enable-vp9 --enable-libyuv --target=arm64-android-gcc --enable-debug --disable-install-docs --log=yes --enable-internal-stats --disable-unit-tests --disable-docs --disable-tools --sdk-path={$HOME}/android-ndk-r14b --extra-cflags="-mfloat-abi=softfp -mfpu=neon -D__STDC_LIMIT_MACROS -Wno-extern-c-compat -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS" --extra-cxxflags="-D__STDC_LIMIT_MACROS -Wno-extern-c-compat" --disable-webm-io

