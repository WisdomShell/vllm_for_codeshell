# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2023 CTranslate2, and Michael Feil
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Inference-only CodeShell model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
import math
from torch import nn
# from transformers import CodeShellConfig
from vllm.transformers_utils.configs.codeshell import CodeShellConfig


from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.model_executor.layers.rotary_embedding import get_rope

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CodeShellAttention(nn.Module):

    def __init__(
        self,
        config: CodeShellConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        # print("Total num heads: ", total_num_heads)
        self.tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % self.tensor_model_parallel_world_size == 0

        self.group_query_attention = config.group_query_attention

        self.num_heads = (total_num_heads //
                          self.tensor_model_parallel_world_size)
        self.head_dim = self.hidden_size // total_num_heads
        
        self.scaling = 1 / math.sqrt(self.head_dim)
        self.rope_scaling = config.rope_scaling
        self.rope_theta = 10000
        self.max_position_embeddings = config.max_position_embeddings

        self.total_num_kv_heads = config.num_query_groups
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tensor_model_parallel_world_size)
        self.kv_dim = self.head_dim * self.num_kv_heads


        self.c_attn = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )

        # self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            linear_method=linear_method,
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=None,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads)

    def butterfly_transform(self, obj: torch.Tensor) -> torch.Tensor:
        query_group = self.num_kv_heads
        n_tokens = obj.shape[0]


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.split(
            [
                self.hidden_size // self.tensor_model_parallel_world_size,
                self.kv_dim, self.kv_dim
            ],
            dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)

        key_cache, value_cache = kv_cache
        attn_output = self.attn(query=q, key=k, value=v, key_cache=key_cache, value_cache=value_cache,
                                input_metadata=input_metadata)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class CodeShellMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: CodeShellConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            linear_method=linear_method,
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.activation_function, quant_config,
                              intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class CodeShellBlock(nn.Module):

    def __init__(
        self,
        config: CodeShellConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (config.n_inner if config.n_inner is not None else 4 *
                     hidden_size)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CodeShellAttention(config, linear_method)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = CodeShellMLP(inner_dim, config, linear_method)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class CodeShellModel(nn.Module):

    def __init__(
        self,
        config: CodeShellConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention

        self.embed_dim = config.hidden_size

        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([
            CodeShellBlock(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(hidden_states=hidden_states, positions=position_ids, kv_cache=kv_caches[i], input_metadata=input_metadata)
        

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class CodeShellForCausalLM(nn.Module):

    def __init__(
        self,
        config: CodeShellConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.transformer = CodeShellModel(config, linear_method)
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "lm_head.weight" in name:
                param = params_dict['lm_head_weight']
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                continue
            if ".attn.bias" in name:
                continue
            if "rotary_emb" in name:
                continue
                
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
