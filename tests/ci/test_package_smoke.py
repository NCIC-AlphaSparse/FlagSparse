# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only smoke tests for CI packaging and public exports."""

import flagsparse


def test_package_version_is_exposed():
    assert flagsparse.__version__ == "1.0.0"


def test_public_exports_are_listed():
    exported = set(dir(flagsparse))
    assert "flagsparse_spmv_csr" in exported
    assert "flagsparse_spmv_csc" in exported
    assert "flagsparse_spmv_bsr" in exported
    assert "flagsparse_spmm_bsr" in exported
    assert "prepare_spmv_csc" in exported
    assert "prepare_spmv_bsr" in exported
    assert "prepare_spmm_bsr_route" in exported
    assert "create_csr_matrix" in exported
    assert "read_mtx_file" in exported
