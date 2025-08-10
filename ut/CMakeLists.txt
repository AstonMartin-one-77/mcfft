
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
    cmocka
    GIT_REPOSITORY https://gitlab.com/cmocka/cmocka.git
    GIT_TAG cmocka-1.1.7
    GIT_SHALLOW    True)
FetchContent_MakeAvailable(cmocka)
set_property(TARGET cmocka PROPERTY C_EXTENSIONS ON)
add_cmocka_test(cmocka_mcfft SOURCES cmocka_mcfft.c
                LINK_LIBRARIES cmocka::cmocka ${PROJECT_NAME})

if(BUILD_BENCHMARKS)
    FetchContent_Declare(
        pffft
        GIT_REPOSITORY https://bitbucket.org/jpommier/pffft.git
        GIT_TAG master
        GIT_SHALLOW True)
    FetchContent_MakeAvailable(pffft)
    add_library(pffft ${pffft_SOURCE_DIR}/pffft.c)
    target_include_directories(pffft PUBLIC ${pffft_SOURCE_DIR})
    set_property(TARGET pffft PROPERTY C_EXTENSIONS ON)
    add_cmocka_test(cmocka_benchmarks_pffft SOURCES cmocka_benchmarks_pffft.c
                    LINK_LIBRARIES cmocka::cmocka pffft ${PROJECT_NAME})
endif()
