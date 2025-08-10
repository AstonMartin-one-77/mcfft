rm -fr ./build_benchmarks
cmake -S . -B ./build_benchmarks -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ./build_benchmarks --config Release
