rm -fr ./build_ut
# -DBUILD_SHARED_LIBS=OFF required to avoid problems with Windows
cmake -S . -B ./build_ut -DBUILD_UT=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ./build_ut --config Release