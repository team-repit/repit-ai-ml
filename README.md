# repit-ai-ml
repit-AI-ML

gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! xvimagesink

General configuration for OpenCV 4.8.0 =====================================
  Version control:               4.8.0-1-g6371ee1

  Platform:
    Timestamp:                   2023-08-29T05:11:38Z
    Host:                        Linux 5.15.111-tegra aarch64
    CMake:                       3.22.1
    CMake generator:             Unix Makefiles
    CMake build tool:            /usr/bin/gmake
    Configuration:               Release

  CPU/HW features:
    Baseline:                    NEON FP16

  C/C++:
    Built as dynamic libs?:      YES
    C++ standard:                11
    C++ Compiler:                /usr/bin/c++  (ver 11.4.0)
    C++ flags (Release):         -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections    -fvisibility=hidden -fvisibility-inlines-hidden -fstack-protector-strong -fPIC -O3 -DNDEBUG  -DNDEBUG -D_FORTIFY_SOURCE=2
    C++ flags (Debug):           -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections    -fvisibility=hidden -fvisibility-inlines-hidden -fstack-protector-strong -fPIC -g  -O0 -DDEBUG -D_DEBUG
    C Compiler:                  /usr/bin/cc
    C flags (Release):           -fsigned-char -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections    -fvisibility=hidden -fstack-protector-strong -fPIC -O3 -DNDEBUG  -DNDEBUG -D_FORTIFY_SOURCE=2
    C flags (Debug):             -fsigned-char -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections    -fvisibility=hidden -fstack-protector-strong -fPIC -g  -O0 -DDEBUG -D_DEBUG
    Linker flags (Release):      -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  -z noexecstack -z relro -z now  
    Linker flags (Debug):        -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  -z noexecstack -z relro -z now  
    ccache:                      NO
    Precompiled headers:         NO
    Extra dependencies:          dl m pthread rt
    3rdparty dependencies:

  OpenCV modules:
    To be built:                 calib3d core dnn features2d flann gapi highgui imgcodecs imgproc ml objdetect photo python3 stitching ts video videoio
    Disabled:                    world
    Disabled by dependency:      -
    Unavailable:                 java python2
    Applications:                tests perf_tests examples apps
    Documentation:               NO
    Non-free algorithms:         NO

  GUI:                           GTK2
    GTK+:                        YES (ver 2.24.33)
      GThread :                  YES (ver 2.72.4)
      GtkGlExt:                  NO

  Media I/O: 
    ZLib:                        /usr/lib/aarch64-linux-gnu/libz.so (ver 1.2.11)
    JPEG:                        /usr/lib/aarch64-linux-gnu/libjpeg.so (ver 80)
    WEBP:                        build (ver encoder: 0x020f)
    PNG:                         /usr/lib/aarch64-linux-gnu/libpng.so (ver 1.6.37)
    TIFF:                        /usr/lib/aarch64-linux-gnu/libtiff.so (ver 42 / 4.3.0)
    JPEG 2000:                   build (ver 2.5.0)
    HDR:                         YES
    SUNRASTER:                   YES
    PXM:                         YES
    PFM:                         YES

  Video I/O:
    FFMPEG:                      YES
      avcodec:                   YES (58.134.100)
      avformat:                  YES (58.76.100)
      avutil:                    YES (56.70.100)
      swscale:                   YES (5.9.100)
      avresample:                NO
    GStreamer:                   YES (1.20.3)
    v4l/v4l2:                    YES (linux/videodev2.h)

  Parallel framework:            TBB (ver 2021.5 interface 12050)

  Trace:                         YES (with Intel ITT)

  Other third-party libraries:
    Lapack:                      NO
    Eigen:                       YES (ver 3.4.0)
    Custom HAL:                  YES (carotene (ver 0.0.1))
    Protobuf:                    build (3.19.1)
    Flatbuffers:                 builtin/3rdparty (23.5.9)

  Python 2:
    Interpreter:                 /usr/bin/python2.7 (ver 2.7.18)
    Libraries:                   NO
    numpy:                       NO (Python wrappers can not be generated)
    install path:                -

  Python 3:
    Interpreter:                 /usr/bin/python3 (ver 3.10.12)
    Libraries:                   /usr/lib/aarch64-linux-gnu/libpython3.10.so (ver 3.10.12)
    numpy:                       /usr/lib/python3/dist-packages/numpy/core/include (ver 1.21.5)
    install path:                lib/python3.10/dist-packages/cv2/python-3.10

  Python (for build):            /usr/bin/python2.7

  Java:                          
    ant:                         NO
    Java:                        NO
    JNI:                         NO
    Java wrappers:               NO
    Java tests:                  NO

  Install to:                    /usr/local
-----------------------------------------------------------------
