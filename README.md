# mcfft
Mixed Complex FFT
## Introduction
Main goal of repository to demostrate mixed (2/4-radix) FFT library and to get optimisations for modern platforms.
PFFFT library is used as performance reference: https://bitbucket.org/jpommier/pffft/src/master
## Getting started
`git clone git@github.com:AstonMartin-one-77/mcfft.git`
### To build library only
`./build.sh`
### To build UTs along with library
`./build_ut.sh`

`cd build_ut/`

`ctest`
### To build Benchmarks along with library
NOTE: Linux ONLY. PFFFT will be fetched via cmake as reference

`./build_benchmarks.sh`

`./build_benchmarks/ut/cmocka_benchmarks_pffft`
### Additional options to add via CMake command line
 * FORCE_EXCLUDE_MALLOC=ON - to exclude usage of malloc/free if it is not needed/supported (NOTE: malloc/free required for benchmarks)
 * FORCE_NEON=ON - to force using NEON optimisations with compile option: -march=armv8-a+simd (NOT MSVC)
 * FORCE_AVX=ON - to force using AVX2 optimisations with compile option: /arch:AVX2 (MSVC) OR -march=haswell -mfma -mavx2 (NOT MSVC)
### Tested platforms
| Platforms      | Ubuntu-22.04 | Windows (MSVC)  | MacOS |
|----------------|--------------|-----------------|-------|
| Intel-12th gen | Yes          | Yes             | No    |
| RPi4           | Yes          | No              | No    |
| Apple M2       | No           | No              | Yes   |

### Benchmark results
NOTE: current version is expected to be a bit slower then PFFFT library (NEON version is slower because of 128-bit SIMD vs 256-bit SIMD used on Intel)
#### 12th Gen Intel® Core™ i3-12100
`./build_benchmarks/ut/cmocka_benchmarks_pffft` 

[==========] utests: Running 5 test(s).

[ RUN      ] cmocka_fft_benchmark_64

Mono time: 7462 Nsec (min: 7425), PFFFT time: 7149 Nsec (min: 6999)

[ RUN      ] cmocka_fft_benchmark_128

Mono time: 13727 Nsec (min: 13638), PFFFT time: 14421 Nsec (min: 14042)

[ RUN      ] cmocka_fft_benchmark_256

Mono time: 30854 Nsec (min: 30746), PFFFT time: 28238 Nsec (min: 28064)

[ RUN      ] cmocka_fft_benchmark_512
#### RPi4: Arm64 Cortex-A53
`./build_benchmarks/ut/cmocka_benchmarks_pffft` 

[==========] utests: Running 5 test(s).

[ RUN      ] cmocka_fft_benchmark_64

Mono time: 64548 Nsec (min: 64466), PFFFT time: 58272 Nsec (min: 58262)

[ RUN      ] cmocka_fft_benchmark_128

Mono time: 151845 Nsec (min: 151601), PFFFT time: 129448 Nsec (min: 128061)

[ RUN      ] cmocka_fft_benchmark_256

Mono time: 295315 Nsec (min: 294496), PFFFT time: 261134 Nsec (min: 260382)

[ RUN      ] cmocka_fft_benchmark_512

Mono time: 698810 Nsec (min: 691437), PFFFT time: 617460 Nsec (min: 612228)

[ RUN      ] cmocka_fft_benchmark_1024

Mono time: 1399608 Nsec (min: 1381980), PFFFT time: 1256493 Nsec (min: 1250284)
