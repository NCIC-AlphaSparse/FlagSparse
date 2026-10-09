# C API backend test profiles

Each file maps one Python/Triton backend suite to its CMake `BACKEND` value and
states whether the C API is currently buildable.  `ctest` test sources remain
shared by operator; configuration includes exactly one profile and labels every
registered test with that profile name. Use `ctest -L capi` for only the C API
tests, or add a profile label when you want to include the Python suites too.

Only CUDA and MUSA are currently C API-buildable. For MUSA, the delivery split
runner uses Python for accuracy and the C API for performance; a pure C API
run is available with the commands below. The other profile files are
deliberate status declarations, so their operator tests must use the Python
backend runner until an adaptor is enabled.

```bash
# MUSA C API
cmake -S capi -B capi/build-musa -G Ninja \
  -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa \
  -DCMAKE_BUILD_TYPE=Release
cmake --build capi/build-musa -j
ctest --test-dir capi/build-musa -L capi --output-on-failure

# Python-only backends: rocm, maca, ascend, xpu, iluvatar
python3 tools/run_backend_tests.py \
  --backend rocm --phase both --mode normal
```
