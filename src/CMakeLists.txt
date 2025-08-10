
if(USE_NEON)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MC_SELECTOR=neon)
    set(SIMD_SRC aarch64/mcfft_neon.c)
elseif(USE_AVX)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MC_SELECTOR=avx)
    set(SIMD_SRC x86/mcfft_avx.c)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${PROJECT_NAME} PRIVATE ${SIMD_SRC} mcfft.c generic/mcfft_generic.c utils.c)
