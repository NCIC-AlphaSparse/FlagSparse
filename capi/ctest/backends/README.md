# C API backend test profiles

Each file maps one Python/Triton backend suite to its CMake `BACKEND` value and
states whether the C API is currently buildable.  `ctest` test sources remain
shared by operator; configuration includes exactly one profile and labels every
registered test with that profile name.  Use `ctest -L cuda` or `ctest -L musa`
after a successful build.

Only CUDA and MUSA are currently C API-buildable.  The other profile files are
deliberate status declarations, so enabling an adaptor later requires an
explicit profile change before C API cases begin to run.
