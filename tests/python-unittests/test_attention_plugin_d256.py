# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Attention plugin vs numpy for head_size=256, fixed batch_size=4.

Exercises the same TensorRT AttentionPlugin path as test_attention_plugin.py
with D=256 (e.g. CuTe DSL FMHA fmha_d256 when ENABLE_CUTE_DSL_FMHA=ON on SM100+).

Requirements:
    - libNvInfer_edgellm_plugin.so (build with matching FMHA options)
    - pycuda, numpy, tensorrt

Usage:
    cd <build-or-repo-root-with-plugin>
    python3 -m pytest third_party/TensorRT-Edge-LLM/tests/python-unittests/test_attention_plugin_d256.py -v
"""

import numpy as np
import pytest
import test_attention_utils as utils

from test_attention_plugin import (
    DEPENDENCIES_AVAILABLE,
    IMPORT_ERROR,
    TestAttentionPluginVsNumpy,
)


@pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"Required dependencies not available: {IMPORT_ERROR}")
class TestAttentionPluginVsNumpyD256(TestAttentionPluginVsNumpy):
    """D=256 plugin checks; profile caps at batch 4 (max_batch_size=4)."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.params = utils.AttentionParams(
            batch_size=4,
            seq_len=1,
            num_q_heads=8,
            num_kv_heads=8,
            head_size=256,
            kv_cache_capacity=64,
            max_batch_size=4,
            max_seq_len=8,
            max_position_embeddings=64,
        )

    def test_plugin_vs_numpy_prefill_batch4(self, request):
        """Prefill: 3 rounds, seq_len=max_seq_len, batch_size=4."""
        self._run_plugin_attention_test(
            num_rounds=3,
            seq_len=self.params.max_seq_len,
            batch_size=4,
            is_prefill=True,
            request=request,
        )

    def test_plugin_vs_numpy_decode_batch4(self, request):
        """Decode: 5 rounds, seq_len=1, batch_size=4."""
        self._run_plugin_attention_test(
            num_rounds=5,
            seq_len=1,
            batch_size=4,
            is_prefill=False,
            request=request,
        )
