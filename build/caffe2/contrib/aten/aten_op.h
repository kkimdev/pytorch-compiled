#pragma once
#include <unordered_map>
#include <string>
#include <ATen/ATen.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/math.h>
#include <iostream>

// a map from descriptor strings (see [DESCRIPTORS])
// to the key in the switch statement that implements them
static std::unordered_map<std::string, int> op_to_key = {
  { "_cast_Byte-non_blocking-1", 0 },
  { "_cast_Byte-1", 1 },
  { "_cast_Char-non_blocking-1", 2 },
  { "_cast_Char-1", 3 },
  { "_cast_Double-non_blocking-1", 4 },
  { "_cast_Double-1", 5 },
  { "_cast_Float-non_blocking-1", 6 },
  { "_cast_Float-1", 7 },
  { "_cast_Int-non_blocking-1", 8 },
  { "_cast_Int-1", 9 },
  { "_cast_Long-non_blocking-1", 10 },
  { "_cast_Long-1", 11 },
  { "_cast_Short-non_blocking-1", 12 },
  { "_cast_Short-1", 13 },
  { "_cast_Half-non_blocking-1", 14 },
  { "_cast_Half-1", 15 },
  { "data-1", 16 },
  { "is_leaf-1", 17 },
  { "output_nr-1", 18 },
  { "_version-1", 19 },
  { "align_as-2", 20 },
  { "align_tensors-*", 21 },
  { "_use_cudnn_ctc_loss-blank-input_lengths-target_lengths-2", 22 },
  { "_cudnn_ctc_loss-blank-deterministic-input_lengths-target_lengths-zero_infinity-2", 23 },
  { "_cudnn_rnn_flatten_weight-batch_first-bidirectional-hidden_size-input_size-mode-num_layers-weight_stride0-*", 24 },
  { "_cudnn_rnn-batch_first-batch_sizes-bidirectional-dropout-hidden_size-mode-num_layers-train-weight_stride0-*", 25 },
  { "_debug_has_internal_overlap-1", 26 },
  { "_fused_dropout-p-1", 27 },
  { "_masked_scale-scale-2", 28 },
  { "_reshape_from_tensor-2", 29 },
  { "_shape_as_tensor-1", 30 },
  { "dropout-p-train-1", 31 },
  { "feature_dropout-p-train-1", 32 },
  { "alpha_dropout-p-train-1", 33 },
  { "feature_alpha_dropout-p-train-1", 34 },
  { "abs-1", 35 },
  { "angle-1", 36 },
  { "real-1", 37 },
  { "imag-1", 38 },
  { "conj-1", 39 },
  { "acos-1", 40 },
  { "avg_pool1d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 41 },
  { "avg_pool1d-ceil_mode-kernel_size-padding-stride-1", 42 },
  { "avg_pool1d-kernel_size-padding-stride-1", 43 },
  { "avg_pool1d-kernel_size-stride-1", 44 },
  { "avg_pool1d-kernel_size-1", 45 },
  { "adaptive_avg_pool1d-output_size-1", 46 },
  { "adaptive_max_pool1d-output_size-1", 47 },
  { "add-alpha-2", 48 },
  { "add-2", 49 },
  { "add-alpha-other-1", 50 },
  { "add-other-1", 51 },
  { "addmv-alpha-beta-3", 52 },
  { "addmv-beta-3", 53 },
  { "addmv-3", 54 },
  { "addr-alpha-beta-3", 55 },
  { "addr-beta-3", 56 },
  { "addr-3", 57 },
  { "affine_grid_generator-align_corners-size-1", 58 },
  { "affine_grid_generator_backward-align_corners-size-1", 59 },
  { "all-dim-keepdim-1", 60 },
  { "all-dim-1", 61 },
  { "allclose-atol-equal_nan-rtol-2", 62 },
  { "allclose-atol-rtol-2", 63 },
  { "allclose-rtol-2", 64 },
  { "allclose-2", 65 },
  { "any-dim-keepdim-1", 66 },
  { "any-dim-1", 67 },
  { "_dim_arange-dim-1", 68 },
  { "argmax-1", 69 },
  { "argmin-1", 70 },
  { "as_strided-size-stride-1", 71 },
  { "asin-1", 72 },
  { "atan-1", 73 },
  { "baddbmm-alpha-beta-3", 74 },
  { "baddbmm-beta-3", 75 },
  { "baddbmm-3", 76 },
  { "batch_norm-cudnn_enabled-eps-momentum-training-5", 77 },
  { "quantized_batch_norm-eps-output_scale-output_zero_point-5", 78 },
  { "_batch_norm_impl_index-cudnn_enabled-eps-momentum-training-5", 79 },
  { "_batch_norm_impl_index_backward-eps-impl_index-output_mask-train-8", 80 },
  { "bernoulli-1", 81 },
  { "bernoulli-p-1", 82 },
  { "bilinear-4", 83 },
  { "binary_cross_entropy-reduction-3", 84 },
  { "binary_cross_entropy-3", 85 },
  { "binary_cross_entropy-2", 86 },
  { "binary_cross_entropy_backward-reduction-4", 87 },
  { "binary_cross_entropy_backward-4", 88 },
  { "binary_cross_entropy_backward-3", 89 },
  { "binary_cross_entropy_with_logits-reduction-4", 90 },
  { "binary_cross_entropy_with_logits-4", 91 },
  { "binary_cross_entropy_with_logits-3", 92 },
  { "binary_cross_entropy_with_logits-2", 93 },
  { "binary_cross_entropy_with_logits_backward-reduction-5", 94 },
  { "binary_cross_entropy_with_logits_backward-5", 95 },
  { "binary_cross_entropy_with_logits_backward-4", 96 },
  { "binary_cross_entropy_with_logits_backward-3", 97 },
  { "bincount-minlength-2", 98 },
  { "bincount-2", 99 },
  { "bincount-1", 100 },
  { "bitwise_not-1", 101 },
  { "logical_not-1", 102 },
  { "logical_xor-2", 103 },
  { "logical_and-2", 104 },
  { "logical_or-2", 105 },
  { "bmm-2", 106 },
  { "broadcast_tensors-*", 107 },
  { "cat-dim-*", 108 },
  { "cat-*", 109 },
  { "ceil-1", 110 },
  { "chain_matmul-*", 111 },
  { "chunk-chunks-dim-1", 112 },
  { "chunk-chunks-1", 113 },
  { "clamp-1", 114 },
  { "clamp_max-max-1", 115 },
  { "clamp_min-min-1", 116 },
  { "cudnn_is_acceptable-1", 117 },
  { "constant_pad_nd-pad-value-1", 118 },
  { "constant_pad_nd-pad-1", 119 },
  { "contiguous-1", 120 },
  { "convolution-dilation-groups-output_padding-padding-stride-transposed-3", 121 },
  { "convolution_overrideable-dilation-groups-output_padding-padding-stride-transposed-3", 122 },
  { "convolution_backward_overrideable-dilation-groups-output_mask-output_padding-padding-stride-transposed-3", 123 },
  { "_convolution-benchmark-cudnn_enabled-deterministic-dilation-groups-output_padding-padding-stride-transposed-3", 124 },
  { "_convolution_nogroup-dilation-output_padding-padding-stride-transposed-3", 125 },
  { "_convolution_double_backward-benchmark-cudnn_enabled-deterministic-dilation-groups-output_mask-output_padding-padding-stride-transposed-6", 126 },
  { "conv1d-dilation-groups-padding-stride-3", 127 },
  { "conv1d-dilation-padding-stride-3", 128 },
  { "conv1d-padding-stride-3", 129 },
  { "conv1d-stride-3", 130 },
  { "conv1d-3", 131 },
  { "conv1d-2", 132 },
  { "conv2d-dilation-groups-padding-stride-3", 133 },
  { "conv2d-dilation-padding-stride-3", 134 },
  { "conv2d-padding-stride-3", 135 },
  { "conv2d-stride-3", 136 },
  { "conv2d-3", 137 },
  { "conv2d-2", 138 },
  { "conv3d-dilation-groups-padding-stride-3", 139 },
  { "conv3d-dilation-padding-stride-3", 140 },
  { "conv3d-padding-stride-3", 141 },
  { "conv3d-stride-3", 142 },
  { "conv3d-3", 143 },
  { "conv3d-2", 144 },
  { "conv_tbc-pad-3", 145 },
  { "conv_tbc-3", 146 },
  { "conv_tbc_backward-pad-4", 147 },
  { "conv_transpose1d-dilation-groups-output_padding-padding-stride-3", 148 },
  { "conv_transpose1d-groups-output_padding-padding-stride-3", 149 },
  { "conv_transpose1d-output_padding-padding-stride-3", 150 },
  { "conv_transpose1d-padding-stride-3", 151 },
  { "conv_transpose1d-stride-3", 152 },
  { "conv_transpose1d-3", 153 },
  { "conv_transpose1d-2", 154 },
  { "conv_transpose2d-dilation-groups-output_padding-padding-stride-3", 155 },
  { "conv_transpose2d-groups-output_padding-padding-stride-3", 156 },
  { "conv_transpose2d-output_padding-padding-stride-3", 157 },
  { "conv_transpose2d-padding-stride-3", 158 },
  { "conv_transpose2d-stride-3", 159 },
  { "conv_transpose2d-3", 160 },
  { "conv_transpose2d-2", 161 },
  { "conv_transpose3d-dilation-groups-output_padding-padding-stride-3", 162 },
  { "conv_transpose3d-groups-output_padding-padding-stride-3", 163 },
  { "conv_transpose3d-output_padding-padding-stride-3", 164 },
  { "conv_transpose3d-padding-stride-3", 165 },
  { "conv_transpose3d-stride-3", 166 },
  { "conv_transpose3d-3", 167 },
  { "conv_transpose3d-2", 168 },
  { "_copy_from-non_blocking-2", 169 },
  { "_copy_from-2", 170 },
  { "cos-1", 171 },
  { "cosh-1", 172 },
  { "cosine_embedding_loss-margin-reduction-3", 173 },
  { "cosine_embedding_loss-margin-3", 174 },
  { "cosine_embedding_loss-3", 175 },
  { "cudnn_affine_grid_generator-C-H-N-W-1", 176 },
  { "cudnn_affine_grid_generator_backward-C-H-N-W-1", 177 },
  { "cudnn_batch_norm-epsilon-exponential_average_factor-training-5", 178 },
  { "cudnn_batch_norm_backward-epsilon-8", 179 },
  { "cudnn_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 180 },
  { "cudnn_convolution-benchmark-deterministic-dilation-groups-padding-stride-2", 181 },
  { "cudnn_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 182 },
  { "cudnn_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 183 },
  { "cudnn_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 184 },
  { "cudnn_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-3", 185 },
  { "cudnn_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-2", 186 },
  { "cudnn_convolution_transpose_backward-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride-3", 187 },
  { "cudnn_convolution_transpose_backward_input-benchmark-deterministic-dilation-groups-padding-stride-2", 188 },
  { "cudnn_convolution_transpose_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 189 },
  { "cudnn_grid_sampler-2", 190 },
  { "cudnn_grid_sampler_backward-3", 191 },
  { "cummax-dim-1", 192 },
  { "cummin-dim-1", 193 },
  { "cumprod-dim-1", 194 },
  { "cumsum-dim-1", 195 },
  { "ctc_loss-blank-input_lengths-reduction-target_lengths-zero_infinity-2", 196 },
  { "ctc_loss-blank-input_lengths-reduction-target_lengths-2", 197 },
  { "ctc_loss-blank-input_lengths-target_lengths-2", 198 },
  { "ctc_loss-input_lengths-target_lengths-2", 199 },
  { "ctc_loss-blank-reduction-zero_infinity-4", 200 },
  { "ctc_loss-blank-reduction-4", 201 },
  { "ctc_loss-blank-4", 202 },
  { "ctc_loss-4", 203 },
  { "_ctc_loss-blank-input_lengths-target_lengths-zero_infinity-2", 204 },
  { "_ctc_loss-blank-input_lengths-target_lengths-2", 205 },
  { "_ctc_loss-input_lengths-target_lengths-2", 206 },
  { "_ctc_loss_backward-blank-input_lengths-target_lengths-zero_infinity-5", 207 },
  { "_ctc_loss_backward-blank-input_lengths-target_lengths-5", 208 },
  { "det-1", 209 },
  { "diag_embed-dim1-dim2-offset-1", 210 },
  { "diag_embed-dim1-offset-1", 211 },
  { "diag_embed-offset-1", 212 },
  { "diag_embed-1", 213 },
  { "diagflat-offset-1", 214 },
  { "diagflat-1", 215 },
  { "diagonal-dim1-dim2-offset-1", 216 },
  { "diagonal-dim1-offset-1", 217 },
  { "diagonal-offset-1", 218 },
  { "diagonal-1", 219 },
  { "div-2", 220 },
  { "div-other-1", 221 },
  { "dot-2", 222 },
  { "embedding-padding_idx-scale_grad_by_freq-sparse-2", 223 },
  { "embedding-padding_idx-scale_grad_by_freq-2", 224 },
  { "embedding-padding_idx-2", 225 },
  { "embedding-2", 226 },
  { "embedding_backward-num_weights-padding_idx-scale_grad_by_freq-sparse-2", 227 },
  { "embedding_dense_backward-num_weights-padding_idx-scale_grad_by_freq-2", 228 },
  { "embedding_sparse_backward-num_weights-padding_idx-scale_grad_by_freq-2", 229 },
  { "embedding_bag-include_last_offset-mode-scale_grad_by_freq-sparse-4", 230 },
  { "embedding_bag-mode-scale_grad_by_freq-sparse-4", 231 },
  { "embedding_bag-mode-scale_grad_by_freq-sparse-3", 232 },
  { "embedding_bag-mode-scale_grad_by_freq-3", 233 },
  { "embedding_bag-scale_grad_by_freq-3", 234 },
  { "embedding_bag-3", 235 },
  { "_embedding_bag-include_last_offset-mode-scale_grad_by_freq-sparse-4", 236 },
  { "_embedding_bag-mode-scale_grad_by_freq-sparse-4", 237 },
  { "_embedding_bag-mode-scale_grad_by_freq-sparse-3", 238 },
  { "_embedding_bag-mode-scale_grad_by_freq-3", 239 },
  { "_embedding_bag-scale_grad_by_freq-3", 240 },
  { "_embedding_bag-3", 241 },
  { "_embedding_bag_backward-mode-num_weights-scale_grad_by_freq-sparse-7", 242 },
  { "_embedding_bag_sparse_backward-mode-num_weights-scale_grad_by_freq-6", 243 },
  { "_embedding_bag_dense_backward-mode-num_weights-scale_grad_by_freq-7", 244 },
  { "_embedding_bag_per_sample_weights_backward-mode-5", 245 },
  { "erf-1", 246 },
  { "erfc-1", 247 },
  { "exp-1", 248 },
  { "expm1-1", 249 },
  { "expand-implicit-size-1", 250 },
  { "expand-size-1", 251 },
  { "expand_as-2", 252 },
  { "flatten-end_dim-start_dim-1", 253 },
  { "flatten-start_dim-1", 254 },
  { "flatten-1", 255 },
  { "floor-1", 256 },
  { "floor_divide-2", 257 },
  { "floor_divide-other-1", 258 },
  { "frac-1", 259 },
  { "grid_sampler-align_corners-interpolation_mode-padding_mode-2", 260 },
  { "grid_sampler_2d-align_corners-interpolation_mode-padding_mode-2", 261 },
  { "grid_sampler_2d_backward-align_corners-interpolation_mode-padding_mode-3", 262 },
  { "grid_sampler_3d-align_corners-interpolation_mode-padding_mode-2", 263 },
  { "grid_sampler_3d_backward-align_corners-interpolation_mode-padding_mode-3", 264 },
  { "hinge_embedding_loss-margin-reduction-2", 265 },
  { "hinge_embedding_loss-margin-2", 266 },
  { "hinge_embedding_loss-2", 267 },
  { "ger-2", 268 },
  { "group_norm-cudnn_enabled-eps-num_groups-3", 269 },
  { "group_norm-eps-num_groups-3", 270 },
  { "group_norm-num_groups-3", 271 },
  { "group_norm-num_groups-2", 272 },
  { "group_norm-num_groups-1", 273 },
  { "fft-normalized-signal_ndim-1", 274 },
  { "fft-signal_ndim-1", 275 },
  { "ifft-normalized-signal_ndim-1", 276 },
  { "ifft-signal_ndim-1", 277 },
  { "rfft-normalized-onesided-signal_ndim-1", 278 },
  { "rfft-normalized-signal_ndim-1", 279 },
  { "rfft-signal_ndim-1", 280 },
  { "irfft-normalized-onesided-signal_ndim-signal_sizes-1", 281 },
  { "irfft-normalized-onesided-signal_ndim-1", 282 },
  { "irfft-normalized-signal_ndim-1", 283 },
  { "irfft-signal_ndim-1", 284 },
  { "_fft_with_size-checked_signal_sizes-complex_input-complex_output-inverse-normalized-onesided-output_sizes-signal_ndim-1", 285 },
  { "_cufft_get_plan_cache_size-device_index-0", 286 },
  { "_cufft_get_plan_cache_max_size-device_index-0", 287 },
  { "index-*", 288 },
  { "index_copy-dim-3", 289 },
  { "index_put-accumulate-*", 290 },
  { "index_put-*", 291 },
  { "instance_norm-cudnn_enabled-eps-momentum-use_input_stats-5", 292 },
  { "inverse-1", 293 },
  { "_inverse_helper-1", 294 },
  { "isclose-atol-equal_nan-rtol-2", 295 },
  { "isclose-atol-rtol-2", 296 },
  { "isclose-rtol-2", 297 },
  { "isclose-2", 298 },
  { "isnan-1", 299 },
  { "is_distributed-1", 300 },
  { "is_floating_point-1", 301 },
  { "is_complex-1", 302 },
  { "is_nonzero-1", 303 },
  { "is_same_size-2", 304 },
  { "is_signed-1", 305 },
  { "kl_div-reduction-2", 306 },
  { "kl_div-2", 307 },
  { "kl_div_backward-reduction-3", 308 },
  { "kl_div_backward-3", 309 },
  { "kthvalue-dim-k-keepdim-1", 310 },
  { "kthvalue-dim-k-1", 311 },
  { "kthvalue-k-1", 312 },
  { "layer_norm-cudnn_enable-eps-normalized_shape-3", 313 },
  { "layer_norm-eps-normalized_shape-3", 314 },
  { "layer_norm-normalized_shape-3", 315 },
  { "layer_norm-normalized_shape-2", 316 },
  { "layer_norm-normalized_shape-1", 317 },
  { "native_layer_norm-M-N-eps-3", 318 },
  { "native_layer_norm_backward-M-N-output_mask-5", 319 },
  { "linear-3", 320 },
  { "linear-2", 321 },
  { "mkldnn_linear-3", 322 },
  { "mkldnn_linear-2", 323 },
  { "fbgemm_linear_int8_weight_fp32_activation-weight_scale-weight_zero_point-5", 324 },
  { "fbgemm_linear_int8_weight-weight_scale-weight_zero_point-5", 325 },
  { "fbgemm_pack_gemm_matrix_fp16-1", 326 },
  { "fbgemm_linear_fp16_weight_fp32_activation-3", 327 },
  { "fbgemm_linear_fp16_weight-3", 328 },
  { "fbgemm_pack_quantized_matrix-1", 329 },
  { "fbgemm_pack_quantized_matrix-K-N-1", 330 },
  { "log-1", 331 },
  { "log10-1", 332 },
  { "log1p-1", 333 },
  { "log2-1", 334 },
  { "logdet-1", 335 },
  { "log_softmax-dim-1", 336 },
  { "_log_softmax-dim-half_to_float-1", 337 },
  { "_log_softmax_backward_data-dim-3", 338 },
  { "logsumexp-dim-keepdim-1", 339 },
  { "logsumexp-dim-1", 340 },
  { "margin_ranking_loss-margin-reduction-3", 341 },
  { "margin_ranking_loss-margin-3", 342 },
  { "margin_ranking_loss-3", 343 },
  { "matmul-2", 344 },
  { "matrix_rank-symmetric-tol-1", 345 },
  { "matrix_rank-tol-1", 346 },
  { "matrix_rank-symmetric-1", 347 },
  { "matrix_rank-1", 348 },
  { "matrix_power-n-1", 349 },
  { "max-dim-keepdim-1", 350 },
  { "max-dim-1", 351 },
  { "max_values-dim-keepdim-1", 352 },
  { "max_values-dim-1", 353 },
  { "max_pool1d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 354 },
  { "max_pool1d_with_indices-dilation-kernel_size-padding-stride-1", 355 },
  { "max_pool1d_with_indices-kernel_size-padding-stride-1", 356 },
  { "max_pool1d_with_indices-kernel_size-stride-1", 357 },
  { "max_pool1d_with_indices-kernel_size-1", 358 },
  { "max_pool1d-ceil_mode-dilation-kernel_size-padding-stride-1", 359 },
  { "max_pool1d-dilation-kernel_size-padding-stride-1", 360 },
  { "max_pool1d-kernel_size-padding-stride-1", 361 },
  { "max_pool1d-kernel_size-stride-1", 362 },
  { "max_pool1d-kernel_size-1", 363 },
  { "max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 364 },
  { "max_pool2d-dilation-kernel_size-padding-stride-1", 365 },
  { "max_pool2d-kernel_size-padding-stride-1", 366 },
  { "max_pool2d-kernel_size-stride-1", 367 },
  { "max_pool2d-kernel_size-1", 368 },
  { "mkldnn_max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 369 },
  { "mkldnn_max_pool2d-dilation-kernel_size-padding-stride-1", 370 },
  { "mkldnn_max_pool2d-kernel_size-padding-stride-1", 371 },
  { "mkldnn_max_pool2d-kernel_size-stride-1", 372 },
  { "mkldnn_max_pool2d-kernel_size-1", 373 },
  { "quantized_max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 374 },
  { "quantized_max_pool2d-dilation-kernel_size-padding-stride-1", 375 },
  { "quantized_max_pool2d-kernel_size-padding-stride-1", 376 },
  { "quantized_max_pool2d-kernel_size-stride-1", 377 },
  { "quantized_max_pool2d-kernel_size-1", 378 },
  { "max_pool3d-ceil_mode-dilation-kernel_size-padding-stride-1", 379 },
  { "max_pool3d-dilation-kernel_size-padding-stride-1", 380 },
  { "max_pool3d-kernel_size-padding-stride-1", 381 },
  { "max_pool3d-kernel_size-stride-1", 382 },
  { "max_pool3d-kernel_size-1", 383 },
  { "mean-1", 384 },
  { "mean-dim-keepdim-1", 385 },
  { "mean-dim-1", 386 },
  { "median-dim-keepdim-1", 387 },
  { "median-dim-1", 388 },
  { "min-dim-keepdim-1", 389 },
  { "min-dim-1", 390 },
  { "min_values-dim-keepdim-1", 391 },
  { "min_values-dim-1", 392 },
  { "mkldnn_convolution-dilation-groups-padding-stride-3", 393 },
  { "mkldnn_convolution_backward_input-bias_defined-dilation-groups-padding-self_size-stride-2", 394 },
  { "mkldnn_convolution_backward_weights-bias_defined-dilation-groups-padding-stride-weight_size-2", 395 },
  { "mkldnn_convolution_backward-dilation-groups-output_mask-padding-stride-3", 396 },
  { "miopen_batch_norm-epsilon-exponential_average_factor-training-5", 397 },
  { "miopen_batch_norm_backward-epsilon-7", 398 },
  { "miopen_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 399 },
  { "miopen_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 400 },
  { "miopen_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 401 },
  { "miopen_convolution_backward_bias-1", 402 },
  { "miopen_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 403 },
  { "miopen_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-3", 404 },
  { "miopen_convolution_transpose_backward-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride-3", 405 },
  { "miopen_convolution_transpose_backward_input-benchmark-deterministic-dilation-groups-padding-stride-2", 406 },
  { "miopen_convolution_transpose_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 407 },
  { "miopen_depthwise_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 408 },
  { "miopen_depthwise_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 409 },
  { "miopen_depthwise_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 410 },
  { "miopen_depthwise_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 411 },
  { "miopen_rnn-batch_first-batch_sizes-bidirectional-dropout-hidden_size-mode-num_layers-train-weight_stride0-*", 412 },
  { "mm-2", 413 },
  { "_sparse_mm-2", 414 },
  { "mode-dim-keepdim-1", 415 },
  { "mode-dim-1", 416 },
  { "mode-1", 417 },
  { "mul-2", 418 },
  { "mul-other-1", 419 },
  { "mv-2", 420 },
  { "mvlgamma-p-1", 421 },
  { "narrow_copy-dim-length-start-1", 422 },
  { "narrow-dim-length-start-1", 423 },
  { "native_batch_norm-eps-momentum-training-5", 424 },
  { "batch_norm_stats-eps-1", 425 },
  { "batch_norm_elemt-eps-5", 426 },
  { "batch_norm_gather_stats-count-eps-momentum-5", 427 },
  { "batch_norm_gather_stats_with_counts-counts-eps-momentum-5", 428 },
  { "native_batch_norm_backward-eps-output_mask-train-7", 429 },
  { "batch_norm_backward_reduce-bias_g-input_g-weight_g-5", 430 },
  { "batch_norm_backward_elemt-7", 431 },
  { "batch_norm_update_stats-momentum-3", 432 },
  { "_nnpack_available-0", 433 },
  { "_nnpack_spatial_convolution-padding-stride-3", 434 },
  { "_nnpack_spatial_convolution-padding-3", 435 },
  { "_nnpack_spatial_convolution_backward-output_mask-padding-3", 436 },
  { "_nnpack_spatial_convolution_backward_input-padding-3", 437 },
  { "_nnpack_spatial_convolution_backward_weight-padding-weightsize-2", 438 },
  { "pairwise_distance-eps-keepdim-p-2", 439 },
  { "pairwise_distance-eps-p-2", 440 },
  { "pairwise_distance-p-2", 441 },
  { "pairwise_distance-2", 442 },
  { "cdist-p-2", 443 },
  { "cdist-2", 444 },
  { "_cdist_backward-p-4", 445 },
  { "pdist-p-1", 446 },
  { "pdist-1", 447 },
  { "_pdist_forward-p-1", 448 },
  { "_pdist_forward-1", 449 },
  { "_pdist_backward-p-3", 450 },
  { "cosine_similarity-dim-eps-2", 451 },
  { "cosine_similarity-dim-2", 452 },
  { "cosine_similarity-2", 453 },
  { "permute-dims-1", 454 },
  { "numpy_T-1", 455 },
  { "pixel_shuffle-upscale_factor-1", 456 },
  { "is_pinned-1", 457 },
  { "pin_memory-1", 458 },
  { "pinverse-rcond-1", 459 },
  { "pinverse-1", 460 },
  { "poisson_nll_loss-eps-full-log_input-reduction-2", 461 },
  { "reciprocal-1", 462 },
  { "neg-1", 463 },
  { "repeat-repeats-1", 464 },
  { "repeat_interleave-1", 465 },
  { "repeat_interleave-2", 466 },
  { "repeat_interleave-repeats-1", 467 },
  { "reshape-shape-1", 468 },
  { "_mkldnn_reshape-shape-1", 469 },
  { "reshape_as-2", 470 },
  { "round-1", 471 },
  { "rrelu-lower-training-upper-1", 472 },
  { "rrelu-lower-upper-1", 473 },
  { "rrelu-lower-1", 474 },
  { "rrelu-1", 475 },
  { "relu-1", 476 },
  { "prelu-2", 477 },
  { "prelu_backward-3", 478 },
  { "gelu-1", 479 },
  { "gelu_backward-2", 480 },
  { "hardshrink-lambd-1", 481 },
  { "hardshrink-1", 482 },
  { "hardshrink_backward-lambd-2", 483 },
  { "rsqrt-1", 484 },
  { "select-dim-index-1", 485 },
  { "selu-1", 486 },
  { "celu-alpha-1", 487 },
  { "celu-1", 488 },
  { "sigmoid-1", 489 },
  { "sin-1", 490 },
  { "sinh-1", 491 },
  { "detach-1", 492 },
  { "size-dim-1", 493 },
  { "slice-dim-end-start-step-1", 494 },
  { "slice-dim-end-start-1", 495 },
  { "slice-dim-start-1", 496 },
  { "slice-dim-1", 497 },
  { "slice-1", 498 },
  { "slogdet-1", 499 },
  { "smm-2", 500 },
  { "softmax-dim-1", 501 },
  { "_softmax-dim-half_to_float-1", 502 },
  { "_softmax_backward_data-dim-3", 503 },
  { "split-dim-split_size-1", 504 },
  { "split-split_size-1", 505 },
  { "split_with_sizes-dim-split_sizes-1", 506 },
  { "split_with_sizes-split_sizes-1", 507 },
  { "squeeze-1", 508 },
  { "squeeze-dim-1", 509 },
  { "sspaddmm-alpha-beta-3", 510 },
  { "sspaddmm-beta-3", 511 },
  { "sspaddmm-3", 512 },
  { "stack-dim-*", 513 },
  { "stack-*", 514 },
  { "stft-n_fft-1", 515 },
  { "stride-dim-1", 516 },
  { "sum-1", 517 },
  { "sum-dim-keepdim-1", 518 },
  { "sum-dim-1", 519 },
  { "sum_to_size-size-1", 520 },
  { "sqrt-1", 521 },
  { "square-1", 522 },
  { "std-unbiased-1", 523 },
  { "std-1", 524 },
  { "std-dim-keepdim-unbiased-1", 525 },
  { "std-dim-unbiased-1", 526 },
  { "std-dim-1", 527 },
  { "std_mean-unbiased-1", 528 },
  { "std_mean-1", 529 },
  { "std_mean-dim-keepdim-unbiased-1", 530 },
  { "std_mean-dim-unbiased-1", 531 },
  { "std_mean-dim-1", 532 },
  { "prod-1", 533 },
  { "prod-dim-keepdim-1", 534 },
  { "prod-dim-1", 535 },
  { "t-1", 536 },
  { "tan-1", 537 },
  { "tanh-1", 538 },
  { "tensordot-dims_other-dims_self-2", 539 },
  { "threshold-threshold-value-1", 540 },
  { "threshold_backward-threshold-2", 541 },
  { "transpose-dim0-dim1-1", 542 },
  { "_mkldnn_transpose-dim0-dim1-1", 543 },
  { "one_hot-num_classes-1", 544 },
  { "one_hot-1", 545 },
  { "flip-dims-1", 546 },
  { "roll-dims-shifts-1", 547 },
  { "roll-shifts-1", 548 },
  { "rot90-dims-k-1", 549 },
  { "rot90-k-1", 550 },
  { "rot90-1", 551 },
  { "trapz-dim-2", 552 },
  { "trapz-2", 553 },
  { "trapz-dim-dx-1", 554 },
  { "trapz-dx-1", 555 },
  { "trapz-1", 556 },
  { "_trilinear-expand1-expand2-expand3-sumdim-unroll_dim-3", 557 },
  { "_trilinear-expand1-expand2-expand3-sumdim-3", 558 },
  { "triplet_margin_loss-eps-margin-p-reduction-swap-3", 559 },
  { "triplet_margin_loss-eps-margin-p-swap-3", 560 },
  { "triplet_margin_loss-eps-margin-p-3", 561 },
  { "triplet_margin_loss-margin-p-3", 562 },
  { "triplet_margin_loss-margin-3", 563 },
  { "triplet_margin_loss-3", 564 },
  { "trunc-1", 565 },
  { "type_as-2", 566 },
  { "_has_compatible_shallow_copy_type-2", 567 },
  { "_unique-return_inverse-sorted-1", 568 },
  { "_unique-sorted-1", 569 },
  { "_unique-1", 570 },
  { "unique_dim-dim-return_counts-return_inverse-sorted-1", 571 },
  { "unique_dim-dim-return_inverse-sorted-1", 572 },
  { "unique_dim-dim-sorted-1", 573 },
  { "unique_dim-dim-1", 574 },
  { "unique_consecutive-return_counts-return_inverse-1", 575 },
  { "unique_consecutive-return_inverse-1", 576 },
  { "unique_consecutive-1", 577 },
  { "unique_dim_consecutive-dim-return_counts-return_inverse-1", 578 },
  { "unique_dim_consecutive-dim-return_inverse-1", 579 },
  { "unique_dim_consecutive-dim-1", 580 },
  { "_unique2-return_counts-return_inverse-sorted-1", 581 },
  { "_unique2-return_inverse-sorted-1", 582 },
  { "_unique2-sorted-1", 583 },
  { "_unique2-1", 584 },
  { "_unsafe_view-size-1", 585 },
  { "unsqueeze-dim-1", 586 },
  { "var-unbiased-1", 587 },
  { "var-1", 588 },
  { "var-dim-keepdim-unbiased-1", 589 },
  { "var-dim-unbiased-1", 590 },
  { "var-dim-1", 591 },
  { "var_mean-unbiased-1", 592 },
  { "var_mean-1", 593 },
  { "var_mean-dim-keepdim-unbiased-1", 594 },
  { "var_mean-dim-unbiased-1", 595 },
  { "var_mean-dim-1", 596 },
  { "view_as-2", 597 },
  { "where-3", 598 },
  { "where-1", 599 },
  { "_s_where-3", 600 },
  { "norm_except_dim-dim-pow-1", 601 },
  { "norm_except_dim-pow-1", 602 },
  { "norm_except_dim-1", 603 },
  { "_weight_norm-dim-2", 604 },
  { "_weight_norm-2", 605 },
  { "_weight_norm_cuda_interface-dim-2", 606 },
  { "_weight_norm_cuda_interface-2", 607 },
  { "_weight_norm_cuda_interface_backward-dim-4", 608 },
  { "_weight_norm_differentiable_backward-dim-4", 609 },
  { "_standard_gamma_grad-2", 610 },
  { "_standard_gamma-1", 611 },
  { "_dirichlet_grad-3", 612 },
  { "_sample_dirichlet-1", 613 },
  { "poisson-1", 614 },
  { "native_norm-p-1", 615 },
  { "native_norm-1", 616 },
  { "_sparse_sum-1", 617 },
  { "_sparse_sum-dim-1", 618 },
  { "_sparse_sum_backward-dim-2", 619 },
  { "norm-p-1", 620 },
  { "norm-1", 621 },
  { "frobenius_norm-1", 622 },
  { "frobenius_norm-dim-keepdim-1", 623 },
  { "frobenius_norm-dim-1", 624 },
  { "nuclear_norm-keepdim-1", 625 },
  { "nuclear_norm-1", 626 },
  { "nuclear_norm-dim-keepdim-1", 627 },
  { "nuclear_norm-dim-1", 628 },
  { "clone-1", 629 },
  { "pow-exponent-1", 630 },
  { "sub-alpha-2", 631 },
  { "sub-2", 632 },
  { "sub-alpha-other-1", 633 },
  { "sub-other-1", 634 },
  { "rsub-alpha-2", 635 },
  { "rsub-2", 636 },
  { "rsub-alpha-other-1", 637 },
  { "rsub-other-1", 638 },
  { "_sparse_addmm-alpha-beta-3", 639 },
  { "_sparse_addmm-beta-3", 640 },
  { "_sparse_addmm-3", 641 },
  { "addmm-alpha-beta-3", 642 },
  { "addmm-beta-3", 643 },
  { "addmm-3", 644 },
  { "sparse_mask-2", 645 },
  { "to_dense-1", 646 },
  { "to_dense_backward-2", 647 },
  { "sparse_dim-1", 648 },
  { "_dimI-1", 649 },
  { "dense_dim-1", 650 },
  { "_dimV-1", 651 },
  { "_nnz-1", 652 },
  { "coalesce-1", 653 },
  { "is_coalesced-1", 654 },
  { "_indices-1", 655 },
  { "_values-1", 656 },
  { "indices-1", 657 },
  { "values-1", 658 },
  { "hspmm-2", 659 },
  { "unbind-dim-1", 660 },
  { "unbind-1", 661 },
  { "to_sparse-sparse_dim-1", 662 },
  { "to_sparse-1", 663 },
  { "to_mkldnn-1", 664 },
  { "mkldnn_reorder_conv2d_weight-dilation-groups-padding-stride-1", 665 },
  { "mkldnn_reorder_conv2d_weight-dilation-padding-stride-1", 666 },
  { "mkldnn_reorder_conv2d_weight-padding-stride-1", 667 },
  { "mkldnn_reorder_conv2d_weight-padding-1", 668 },
  { "mkldnn_reorder_conv2d_weight-1", 669 },
  { "to_mkldnn_backward-2", 670 },
  { "dequantize-1", 671 },
  { "q_zero_point-1", 672 },
  { "q_per_channel_scales-1", 673 },
  { "q_per_channel_zero_points-1", 674 },
  { "q_per_channel_axis-1", 675 },
  { "int_repr-1", 676 },
  { "_make_per_tensor_quantized_tensor-scale-zero_point-1", 677 },
  { "_make_per_channel_quantized_tensor-axis-3", 678 },
  { "fake_quantize_per_tensor_affine-quant_max-quant_min-scale-zero_point-1", 679 },
  { "fake_quantize_per_tensor_affine_backward-quant_max-quant_min-scale-zero_point-2", 680 },
  { "fake_quantize_per_channel_affine-axis-quant_max-quant_min-3", 681 },
  { "fake_quantize_per_channel_affine_backward-axis-quant_max-quant_min-4", 682 },
  { "meshgrid-*", 683 },
  { "cartesian_prod-*", 684 },
  { "combinations-r-with_replacement-1", 685 },
  { "combinations-r-1", 686 },
  { "combinations-1", 687 },
  { "item-1", 688 },
  { "_local_scalar_dense-1", 689 },
  { "_thnn_fused_lstm_cell-5", 690 },
  { "_thnn_fused_lstm_cell-4", 691 },
  { "_thnn_fused_lstm_cell-3", 692 },
  { "_thnn_fused_lstm_cell_backward-has_bias-5", 693 },
  { "_thnn_differentiable_lstm_cell_backward-8", 694 },
  { "_thnn_fused_gru_cell-5", 695 },
  { "_thnn_fused_gru_cell-4", 696 },
  { "_thnn_fused_gru_cell-3", 697 },
  { "_thnn_fused_gru_cell_backward-has_bias-2", 698 },
  { "_thnn_differentiable_gru_cell_backward-6", 699 },
  { "lstm-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 700 },
  { "lstm-bidirectional-dropout-has_biases-num_layers-train-*", 701 },
  { "gru-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 702 },
  { "gru-bidirectional-dropout-has_biases-num_layers-train-*", 703 },
  { "rnn_tanh-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 704 },
  { "rnn_tanh-bidirectional-dropout-has_biases-num_layers-train-*", 705 },
  { "rnn_relu-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 706 },
  { "rnn_relu-bidirectional-dropout-has_biases-num_layers-train-*", 707 },
  { "lstm_cell-*", 708 },
  { "gru_cell-6", 709 },
  { "gru_cell-5", 710 },
  { "gru_cell-4", 711 },
  { "rnn_tanh_cell-6", 712 },
  { "rnn_tanh_cell-5", 713 },
  { "rnn_tanh_cell-4", 714 },
  { "rnn_relu_cell-6", 715 },
  { "rnn_relu_cell-5", 716 },
  { "rnn_relu_cell-4", 717 },
  { "quantized_lstm-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 718 },
  { "quantized_lstm-bidirectional-dropout-has_biases-num_layers-train-*", 719 },
  { "quantized_gru-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 720 },
  { "quantized_gru-bidirectional-dropout-has_biases-num_layers-train-*", 721 },
  { "quantized_lstm_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-*", 722 },
  { "quantized_gru_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 723 },
  { "quantized_rnn_relu_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 724 },
  { "quantized_rnn_tanh_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 725 },
  { "_pack_padded_sequence-batch_first-2", 726 },
  { "_pack_padded_sequence_backward-batch_first-input_size-2", 727 },
  { "_pad_packed_sequence-batch_first-padding_value-total_length-2", 728 },
  { "is_set_to-2", 729 },
  { "masked_fill-value-2", 730 },
  { "masked_fill-3", 731 },
  { "masked_scatter-3", 732 },
  { "view-size-1", 733 },
  { "index_add-dim-3", 734 },
  { "index_fill-dim-value-2", 735 },
  { "index_fill-dim-3", 736 },
  { "scatter-dim-3", 737 },
  { "scatter-dim-value-2", 738 },
  { "scatter_add-dim-3", 739 },
  { "bitwise_and-other-1", 740 },
  { "bitwise_and-2", 741 },
  { "__and__-other-1", 742 },
  { "__and__-2", 743 },
  { "bitwise_or-other-1", 744 },
  { "bitwise_or-2", 745 },
  { "__or__-other-1", 746 },
  { "__or__-2", 747 },
  { "bitwise_xor-other-1", 748 },
  { "bitwise_xor-2", 749 },
  { "__xor__-other-1", 750 },
  { "__xor__-2", 751 },
  { "__lshift__-other-1", 752 },
  { "__lshift__-2", 753 },
  { "__rshift__-other-1", 754 },
  { "__rshift__-2", 755 },
  { "addbmm-alpha-beta-3", 756 },
  { "addbmm-beta-3", 757 },
  { "addbmm-3", 758 },
  { "diag-diagonal-1", 759 },
  { "diag-1", 760 },
  { "cross-2", 761 },
  { "triu-diagonal-1", 762 },
  { "triu-1", 763 },
  { "tril-diagonal-1", 764 },
  { "tril-1", 765 },
  { "trace-1", 766 },
  { "ne-other-1", 767 },
  { "ne-2", 768 },
  { "eq-other-1", 769 },
  { "eq-2", 770 },
  { "ge-other-1", 771 },
  { "ge-2", 772 },
  { "le-other-1", 773 },
  { "le-2", 774 },
  { "gt-other-1", 775 },
  { "gt-2", 776 },
  { "lt-other-1", 777 },
  { "lt-2", 778 },
  { "take-2", 779 },
  { "index_select-dim-2", 780 },
  { "masked_select-2", 781 },
  { "nonzero-1", 782 },
  { "nonzero_numpy-1", 783 },
  { "gather-dim-sparse_grad-2", 784 },
  { "gather-dim-2", 785 },
  { "_gather_sparse_backward-dim-3", 786 },
  { "addcmul-value-3", 787 },
  { "addcmul-3", 788 },
  { "addcdiv-value-3", 789 },
  { "addcdiv-3", 790 },
  { "lstsq-2", 791 },
  { "triangular_solve-transpose-unitriangular-upper-2", 792 },
  { "triangular_solve-transpose-upper-2", 793 },
  { "triangular_solve-upper-2", 794 },
  { "triangular_solve-2", 795 },
  { "_triangular_solve_helper-transpose-unitriangular-upper-2", 796 },
  { "symeig-eigenvectors-upper-1", 797 },
  { "symeig-eigenvectors-1", 798 },
  { "symeig-1", 799 },
  { "_symeig_helper-eigenvectors-upper-1", 800 },
  { "eig-eigenvectors-1", 801 },
  { "eig-1", 802 },
  { "svd-compute_uv-some-1", 803 },
  { "svd-some-1", 804 },
  { "svd-1", 805 },
  { "_svd_helper-compute_uv-some-1", 806 },
  { "cholesky-upper-1", 807 },
  { "cholesky-1", 808 },
  { "_cholesky_helper-upper-1", 809 },
  { "cholesky_solve-upper-2", 810 },
  { "cholesky_solve-2", 811 },
  { "_cholesky_solve_helper-upper-2", 812 },
  { "solve-2", 813 },
  { "_solve_helper-2", 814 },
  { "cholesky_inverse-upper-1", 815 },
  { "cholesky_inverse-1", 816 },
  { "qr-some-1", 817 },
  { "qr-1", 818 },
  { "_qr_helper-some-1", 819 },
  { "geqrf-1", 820 },
  { "orgqr-2", 821 },
  { "ormqr-left-transpose-3", 822 },
  { "ormqr-left-3", 823 },
  { "ormqr-3", 824 },
  { "_lu_with_info-check_errors-pivot-1", 825 },
  { "_lu_with_info-pivot-1", 826 },
  { "_lu_with_info-1", 827 },
  { "lu_solve-3", 828 },
  { "_lu_solve_helper-3", 829 },
  { "multinomial-num_samples-replacement-1", 830 },
  { "multinomial-num_samples-1", 831 },
  { "_multinomial_alias_setup-1", 832 },
  { "_multinomial_alias_draw-num_samples-2", 833 },
  { "lgamma-1", 834 },
  { "digamma-1", 835 },
  { "polygamma-n-1", 836 },
  { "erfinv-1", 837 },
  { "sign-1", 838 },
  { "dist-p-2", 839 },
  { "dist-2", 840 },
  { "atan2-2", 841 },
  { "lerp-weight-2", 842 },
  { "lerp-3", 843 },
  { "histc-bins-max-min-1", 844 },
  { "histc-bins-min-1", 845 },
  { "histc-bins-1", 846 },
  { "histc-1", 847 },
  { "fmod-other-1", 848 },
  { "fmod-2", 849 },
  { "remainder-other-1", 850 },
  { "remainder-2", 851 },
  { "min-2", 852 },
  { "min-1", 853 },
  { "max-2", 854 },
  { "max-1", 855 },
  { "median-1", 856 },
  { "sort-descending-dim-1", 857 },
  { "sort-dim-1", 858 },
  { "sort-1", 859 },
  { "argsort-descending-dim-1", 860 },
  { "argsort-dim-1", 861 },
  { "argsort-1", 862 },
  { "topk-dim-k-largest-sorted-1", 863 },
  { "topk-dim-k-largest-1", 864 },
  { "topk-dim-k-1", 865 },
  { "topk-k-1", 866 },
  { "all-1", 867 },
  { "any-1", 868 },
  { "renorm-dim-maxnorm-p-1", 869 },
  { "unfold-dimension-size-step-1", 870 },
  { "equal-2", 871 },
  { "pow-2", 872 },
  { "pow-self-1", 873 },
  { "alias-1", 874 },
  { "_addr-alpha-beta-3", 875 },
  { "_addr-beta-3", 876 },
  { "_addr-3", 877 },
  { "_cumsum-dim-1", 878 },
  { "_cumprod-dim-1", 879 },
  { "_var-unbiased-1", 880 },
  { "_var-1", 881 },
  { "_std-unbiased-1", 882 },
  { "_std-1", 883 },
  { "_cat-dim-*", 884 },
  { "_cat-*", 885 },
  { "_mode-dim-keepdim-1", 886 },
  { "_mode-dim-1", 887 },
  { "_mode-1", 888 },
  { "_max-dim-keepdim-1", 889 },
  { "_max-dim-1", 890 },
  { "_min-dim-keepdim-1", 891 },
  { "_min-dim-1", 892 },
  { "mse_loss-reduction-2", 893 },
  { "mse_loss-2", 894 },
  { "mse_loss_backward-reduction-3", 895 },
  { "l1_loss-reduction-2", 896 },
  { "l1_loss-2", 897 },
  { "l1_loss_backward-reduction-3", 898 },
  { "multi_margin_loss-margin-p-reduction-3", 899 },
  { "multi_margin_loss-margin-p-3", 900 },
  { "multi_margin_loss-margin-p-2", 901 },
  { "multi_margin_loss-p-2", 902 },
  { "multi_margin_loss-2", 903 },
  { "multi_margin_loss_backward-margin-p-reduction-4", 904 },
  { "multi_margin_loss_backward-margin-p-4", 905 },
  { "multi_margin_loss_backward-margin-p-3", 906 },
  { "multilabel_margin_loss-reduction-2", 907 },
  { "multilabel_margin_loss-2", 908 },
  { "multilabel_margin_loss_forward-reduction-2", 909 },
  { "multilabel_margin_loss_backward-reduction-4", 910 },
  { "nll_loss-ignore_index-reduction-3", 911 },
  { "nll_loss-reduction-3", 912 },
  { "nll_loss-3", 913 },
  { "nll_loss-2", 914 },
  { "nll_loss_forward-ignore_index-reduction-3", 915 },
  { "nll_loss_backward-ignore_index-reduction-5", 916 },
  { "nll_loss2d-ignore_index-reduction-3", 917 },
  { "nll_loss2d-reduction-3", 918 },
  { "nll_loss2d-3", 919 },
  { "nll_loss2d-2", 920 },
  { "nll_loss2d_forward-ignore_index-reduction-3", 921 },
  { "nll_loss2d_backward-ignore_index-reduction-5", 922 },
  { "smooth_l1_loss-reduction-2", 923 },
  { "smooth_l1_loss-2", 924 },
  { "smooth_l1_loss_backward-reduction-3", 925 },
  { "soft_margin_loss-reduction-2", 926 },
  { "soft_margin_loss-2", 927 },
  { "soft_margin_loss_backward-reduction-3", 928 },
  { "elu-alpha-input_scale-scale-1", 929 },
  { "elu-alpha-scale-1", 930 },
  { "elu-alpha-1", 931 },
  { "elu-1", 932 },
  { "elu_backward-alpha-input_scale-scale-2", 933 },
  { "glu-dim-1", 934 },
  { "glu-1", 935 },
  { "glu_backward-dim-2", 936 },
  { "hardtanh-max_val-min_val-1", 937 },
  { "hardtanh-min_val-1", 938 },
  { "hardtanh-1", 939 },
  { "hardtanh_backward-max_val-min_val-2", 940 },
  { "leaky_relu-negative_slope-1", 941 },
  { "leaky_relu-1", 942 },
  { "leaky_relu_backward-negative_slope-2", 943 },
  { "log_sigmoid-1", 944 },
  { "log_sigmoid_forward-1", 945 },
  { "log_sigmoid_backward-3", 946 },
  { "rrelu_with_noise-lower-training-upper-2", 947 },
  { "rrelu_with_noise-lower-upper-2", 948 },
  { "rrelu_with_noise-lower-2", 949 },
  { "rrelu_with_noise-2", 950 },
  { "rrelu_with_noise_backward-lower-training-upper-3", 951 },
  { "softplus-beta-threshold-1", 952 },
  { "softplus-beta-1", 953 },
  { "softplus-1", 954 },
  { "softplus_backward-beta-threshold-3", 955 },
  { "softshrink-lambd-1", 956 },
  { "softshrink-1", 957 },
  { "softshrink_backward-lambd-2", 958 },
  { "adaptive_avg_pool2d-output_size-1", 959 },
  { "mkldnn_adaptive_avg_pool2d-output_size-1", 960 },
  { "_adaptive_avg_pool2d-output_size-1", 961 },
  { "_adaptive_avg_pool2d_backward-2", 962 },
  { "adaptive_avg_pool3d-output_size-1", 963 },
  { "adaptive_avg_pool3d_backward-2", 964 },
  { "adaptive_max_pool2d-output_size-1", 965 },
  { "adaptive_max_pool2d_backward-3", 966 },
  { "adaptive_max_pool3d-output_size-1", 967 },
  { "adaptive_max_pool3d_backward-3", 968 },
  { "avg_pool2d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 969 },
  { "avg_pool2d-ceil_mode-kernel_size-padding-stride-1", 970 },
  { "avg_pool2d-kernel_size-padding-stride-1", 971 },
  { "avg_pool2d-kernel_size-stride-1", 972 },
  { "avg_pool2d-kernel_size-1", 973 },
  { "avg_pool3d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 974 },
  { "avg_pool3d-ceil_mode-kernel_size-padding-stride-1", 975 },
  { "avg_pool3d-kernel_size-padding-stride-1", 976 },
  { "avg_pool3d-kernel_size-stride-1", 977 },
  { "avg_pool3d-kernel_size-1", 978 },
  { "fractional_max_pool2d-kernel_size-output_size-2", 979 },
  { "fractional_max_pool2d_backward-kernel_size-output_size-3", 980 },
  { "fractional_max_pool3d-kernel_size-output_size-2", 981 },
  { "fractional_max_pool3d_backward-kernel_size-output_size-3", 982 },
  { "max_pool2d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 983 },
  { "max_pool2d_with_indices-dilation-kernel_size-padding-stride-1", 984 },
  { "max_pool2d_with_indices-kernel_size-padding-stride-1", 985 },
  { "max_pool2d_with_indices-kernel_size-stride-1", 986 },
  { "max_pool2d_with_indices-kernel_size-1", 987 },
  { "max_pool2d_with_indices_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 988 },
  { "max_pool3d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 989 },
  { "max_pool3d_with_indices-dilation-kernel_size-padding-stride-1", 990 },
  { "max_pool3d_with_indices-kernel_size-padding-stride-1", 991 },
  { "max_pool3d_with_indices-kernel_size-stride-1", 992 },
  { "max_pool3d_with_indices-kernel_size-1", 993 },
  { "max_pool3d_with_indices_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 994 },
  { "max_unpool2d-output_size-2", 995 },
  { "max_unpool2d_backward-output_size-3", 996 },
  { "max_unpool3d-output_size-padding-stride-2", 997 },
  { "max_unpool3d_backward-output_size-padding-stride-3", 998 },
  { "reflection_pad1d-padding-1", 999 },
  { "reflection_pad1d_backward-padding-2", 1000 },
  { "reflection_pad2d-padding-1", 1001 },
  { "reflection_pad2d_backward-padding-2", 1002 },
  { "replication_pad1d-padding-1", 1003 },
  { "replication_pad1d_backward-padding-2", 1004 },
  { "replication_pad2d-padding-1", 1005 },
  { "replication_pad2d_backward-padding-2", 1006 },
  { "replication_pad3d-padding-1", 1007 },
  { "replication_pad3d_backward-padding-2", 1008 },
  { "upsample_linear1d-align_corners-output_size-1", 1009 },
  { "upsample_linear1d_backward-align_corners-input_size-output_size-1", 1010 },
  { "upsample_bilinear2d-align_corners-output_size-1", 1011 },
  { "upsample_bilinear2d_backward-align_corners-input_size-output_size-1", 1012 },
  { "upsample_bicubic2d-align_corners-output_size-1", 1013 },
  { "upsample_bicubic2d_backward-align_corners-input_size-output_size-1", 1014 },
  { "upsample_trilinear3d-align_corners-output_size-1", 1015 },
  { "upsample_trilinear3d_backward-align_corners-input_size-output_size-1", 1016 },
  { "upsample_nearest1d-output_size-1", 1017 },
  { "upsample_nearest1d_backward-input_size-output_size-1", 1018 },
  { "upsample_nearest2d-output_size-1", 1019 },
  { "upsample_nearest2d_backward-input_size-output_size-1", 1020 },
  { "upsample_nearest3d-output_size-1", 1021 },
  { "upsample_nearest3d_backward-input_size-output_size-1", 1022 },
  { "sigmoid_backward-2", 1023 },
  { "tanh_backward-2", 1024 },
  { "slow_conv_transpose2d-dilation-kernel_size-output_padding-padding-stride-3", 1025 },
  { "slow_conv_transpose2d-kernel_size-output_padding-padding-stride-3", 1026 },
  { "slow_conv_transpose2d-kernel_size-padding-stride-3", 1027 },
  { "slow_conv_transpose2d-kernel_size-stride-3", 1028 },
  { "slow_conv_transpose2d-kernel_size-3", 1029 },
  { "slow_conv_transpose2d-kernel_size-2", 1030 },
  { "slow_conv_transpose2d_backward-dilation-kernel_size-output_mask-output_padding-padding-stride-5", 1031 },
  { "slow_conv_transpose3d-dilation-kernel_size-output_padding-padding-stride-3", 1032 },
  { "slow_conv_transpose3d-kernel_size-output_padding-padding-stride-3", 1033 },
  { "slow_conv_transpose3d-kernel_size-padding-stride-3", 1034 },
  { "slow_conv_transpose3d-kernel_size-stride-3", 1035 },
  { "slow_conv_transpose3d-kernel_size-3", 1036 },
  { "slow_conv_transpose3d-kernel_size-2", 1037 },
  { "slow_conv_transpose3d_backward-dilation-kernel_size-output_mask-output_padding-padding-stride-5", 1038 },
  { "thnn_conv2d-kernel_size-padding-stride-3", 1039 },
  { "thnn_conv2d-kernel_size-stride-3", 1040 },
  { "thnn_conv2d-kernel_size-3", 1041 },
  { "thnn_conv2d-kernel_size-2", 1042 },
  { "thnn_conv2d_forward-kernel_size-padding-stride-3", 1043 },
  { "thnn_conv2d_backward-kernel_size-output_mask-padding-stride-5", 1044 },
  { "thnn_conv_depthwise2d-dilation-kernel_size-padding-stride-3", 1045 },
  { "thnn_conv_depthwise2d-kernel_size-padding-stride-3", 1046 },
  { "thnn_conv_depthwise2d-kernel_size-stride-3", 1047 },
  { "thnn_conv_depthwise2d-kernel_size-3", 1048 },
  { "thnn_conv_depthwise2d-kernel_size-2", 1049 },
  { "thnn_conv_depthwise2d_forward-dilation-kernel_size-padding-stride-3", 1050 },
  { "thnn_conv_depthwise2d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1051 },
  { "slow_conv3d-kernel_size-padding-stride-3", 1052 },
  { "slow_conv3d-kernel_size-stride-3", 1053 },
  { "slow_conv3d-kernel_size-3", 1054 },
  { "slow_conv3d-kernel_size-2", 1055 },
  { "slow_conv3d_forward-kernel_size-padding-stride-3", 1056 },
  { "slow_conv3d_backward-kernel_size-output_mask-padding-stride-5", 1057 },
  { "slow_conv_dilated2d-dilation-kernel_size-padding-stride-3", 1058 },
  { "slow_conv_dilated2d-kernel_size-padding-stride-3", 1059 },
  { "slow_conv_dilated2d-kernel_size-stride-3", 1060 },
  { "slow_conv_dilated2d-kernel_size-3", 1061 },
  { "slow_conv_dilated2d-kernel_size-2", 1062 },
  { "slow_conv_dilated2d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1063 },
  { "slow_conv_dilated3d-dilation-kernel_size-padding-stride-3", 1064 },
  { "slow_conv_dilated3d-kernel_size-padding-stride-3", 1065 },
  { "slow_conv_dilated3d-kernel_size-stride-3", 1066 },
  { "slow_conv_dilated3d-kernel_size-3", 1067 },
  { "slow_conv_dilated3d-kernel_size-2", 1068 },
  { "slow_conv_dilated3d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1069 },
  { "col2im-dilation-kernel_size-output_size-padding-stride-1", 1070 },
  { "col2im_backward-dilation-kernel_size-padding-stride-1", 1071 },
  { "im2col-dilation-kernel_size-padding-stride-1", 1072 },
  { "im2col_backward-dilation-input_size-kernel_size-padding-stride-1", 1073 },
  { "isfinite-1", 1074 },
  { "isinf-1", 1075 },
};

namespace caffe2 {

using at::Half; // for AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ...)

template <class Context>
class ATenOp : public Operator<Context> {
 public:
  ATenOp(const OperatorDef& operator_def, Workspace* ws)
  : Operator<Context>(operator_def, ws) {
    VLOG(2) << "ATen OpDef: " << ProtoDebugString(operator_def) << "\n";
    switch(findImplementation(operator_def)) {
      case 0: { // _cast_Byte
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Byte(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1: { // _cast_Byte
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Byte(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 2: { // _cast_Char
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Char(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 3: { // _cast_Char
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Char(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 4: { // _cast_Double
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Double(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 5: { // _cast_Double
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Double(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 6: { // _cast_Float
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Float(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 7: { // _cast_Float
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Float(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 8: { // _cast_Int
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Int(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 9: { // _cast_Int
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Int(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 10: { // _cast_Long
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Long(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 11: { // _cast_Long
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Long(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 12: { // _cast_Short
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Short(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 13: { // _cast_Short
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Short(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 14: { // _cast_Half
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Half(self, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 15: { // _cast_Half
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cast_Half(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 16: { // data
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.data();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 17: { // is_leaf
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.is_leaf();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 18: { // output_nr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.output_nr();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 19: { // _version
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self._version();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 20: { // align_as
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.align_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 21: { // align_tensors
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::align_tensors(tensors);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 22: { // _use_cudnn_ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 23: { // _cudnn_ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          bool deterministic = readAttribute<int64_t>("deterministic");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 24: { // _cudnn_rnn_flatten_weight
          int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
          int64_t input_size = readAttribute<int64_t>("input_size");
          int64_t mode = readAttribute<int64_t>("mode");
          int64_t hidden_size = readAttribute<int64_t>("hidden_size");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          bool batch_first = readAttribute<int64_t>("batch_first");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight_arr = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 25: { // _cudnn_rnn
          int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
          int64_t mode = readAttribute<int64_t>("mode");
          int64_t hidden_size = readAttribute<int64_t>("hidden_size");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          bool batch_first = readAttribute<int64_t>("batch_first");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          auto batch_sizes = readIntArrayRef("batch_sizes");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto weight = peekSlice(1, InputSize() - 5, InputSize());
              auto weight_buf = peek(1, 5);
              auto hx = peek(2, 5);
              auto cx = peek(3, 5);
              auto dropout_state = peek(4, 5);
              auto the_result = at::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 26: { // _debug_has_internal_overlap
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_debug_has_internal_overlap(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 27: { // _fused_dropout
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_fused_dropout(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 28: { // _masked_scale
          double scale = readAttribute<float>("scale");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = at::_masked_scale(self, mask, scale);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 29: { // _reshape_from_tensor
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto shape = peek(1, 2);
              auto the_result = at::_reshape_from_tensor(self, shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 30: { // _shape_as_tensor
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_shape_as_tensor(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 31: { // dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 32: { // feature_dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::feature_dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 33: { // alpha_dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::alpha_dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 34: { // feature_alpha_dropout
          double p = readAttribute<float>("p");
          bool train = readAttribute<int64_t>("train");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::feature_alpha_dropout(input, p, train);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 35: { // abs
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::abs(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 36: { // angle
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::angle(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 37: { // real
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::real(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 38: { // imag
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::imag(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 39: { // conj
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::conj(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 40: { // acos
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::acos(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 41: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          bool count_include_pad = readAttribute<int64_t>("count_include_pad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 42: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 43: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 44: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 45: { // avg_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool1d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 46: { // adaptive_avg_pool1d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::adaptive_avg_pool1d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 47: { // adaptive_max_pool1d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::adaptive_max_pool1d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 48: { // add
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::add(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 49: { // add
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::add(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 50: { // add
          at::Scalar other = readScalarAttribute("other");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::add(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 51: { // add
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::add(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 52: { // addmv
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat = peek(1, 3);
              auto vec = peek(2, 3);
              auto the_result = at::addmv(self, mat, vec, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 53: { // addmv
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat = peek(1, 3);
              auto vec = peek(2, 3);
              auto the_result = at::addmv(self, mat, vec, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 54: { // addmv
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat = peek(1, 3);
              auto vec = peek(2, 3);
              auto the_result = at::addmv(self, mat, vec);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 55: { // addr
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::addr(self, vec1, vec2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 56: { // addr
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::addr(self, vec1, vec2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 57: { // addr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::addr(self, vec1, vec2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 58: { // affine_grid_generator
          auto size = readIntArrayRef("size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto theta = peek(0, 1);
              auto the_result = at::affine_grid_generator(theta, size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 59: { // affine_grid_generator_backward
          auto size = readIntArrayRef("size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 1);
              auto the_result = at::affine_grid_generator_backward(grad, size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 60: { // all
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::all(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 61: { // all
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::all(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 62: { // allclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          bool equal_nan = readAttribute<int64_t>("equal_nan");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other, rtol, atol, equal_nan);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 63: { // allclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other, rtol, atol);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 64: { // allclose
          double rtol = readAttribute<float>("rtol");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other, rtol);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 65: { // allclose
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::allclose(self, other);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 66: { // any
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::any(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 67: { // any
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::any(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 68: { // _dim_arange
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto like = peek(0, 1);
              auto the_result = at::_dim_arange(like, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 69: { // argmax
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::argmax(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 70: { // argmin
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::argmin(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 71: { // as_strided
          auto size = readIntArrayRef("size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::as_strided(self, size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 72: { // asin
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::asin(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 73: { // atan
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::atan(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 74: { // baddbmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::baddbmm(self, batch1, batch2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 75: { // baddbmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::baddbmm(self, batch1, batch2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 76: { // baddbmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::baddbmm(self, batch1, batch2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 77: { // batch_norm
          bool training = readAttribute<int64_t>("training");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 78: { // quantized_batch_norm
          double eps = readAttribute<float>("eps");
          double output_scale = readAttribute<float>("output_scale");
          int64_t output_zero_point = readAttribute<int64_t>("output_zero_point");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto mean = peek(3, 5);
              auto var = peek(4, 5);
              auto the_result = at::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 79: { // _batch_norm_impl_index
          bool training = readAttribute<int64_t>("training");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignToValue<int64_t>(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 80: { // _batch_norm_impl_index_backward
          int64_t impl_index = readAttribute<int64_t>("impl_index");
          bool train = readAttribute<int64_t>("train");
          double eps = readAttribute<float>("eps");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 8);
              auto grad_output = peek(1, 8);
              auto weight = peek(2, 8);
              auto running_mean = peek(3, 8);
              auto running_var = peek(4, 8);
              auto save_mean = peek(5, 8);
              auto save_var_transform = peek(6, 8);
              auto reservedSpace = peek(7, 8);
              auto the_result = at::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 81: { // bernoulli
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bernoulli(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 82: { // bernoulli
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bernoulli(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 83: { // bilinear
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 4);
              auto input2 = peek(1, 4);
              auto weight = peek(2, 4);
              auto bias = peek(3, 4);
              auto the_result = at::bilinear(input1, input2, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 84: { // binary_cross_entropy
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::binary_cross_entropy(self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 85: { // binary_cross_entropy
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::binary_cross_entropy(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 86: { // binary_cross_entropy
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::binary_cross_entropy(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 87: { // binary_cross_entropy_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 88: { // binary_cross_entropy_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_backward(grad_output, self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 89: { // binary_cross_entropy_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::binary_cross_entropy_backward(grad_output, self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 90: { // binary_cross_entropy_with_logits
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 4);
              auto target = peek(1, 4);
              auto weight = peek(2, 4);
              auto pos_weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 91: { // binary_cross_entropy_with_logits
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 4);
              auto target = peek(1, 4);
              auto weight = peek(2, 4);
              auto pos_weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_with_logits(self, target, weight, pos_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 92: { // binary_cross_entropy_with_logits
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::binary_cross_entropy_with_logits(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 93: { // binary_cross_entropy_with_logits
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::binary_cross_entropy_with_logits(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 94: { // binary_cross_entropy_with_logits_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto pos_weight = peek(4, 5);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 95: { // binary_cross_entropy_with_logits_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto pos_weight = peek(4, 5);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 96: { // binary_cross_entropy_with_logits_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 97: { // binary_cross_entropy_with_logits_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 98: { // bincount
          int64_t minlength = readAttribute<int64_t>("minlength");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weights = peek(1, 2);
              auto the_result = at::bincount(self, weights, minlength);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 99: { // bincount
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weights = peek(1, 2);
              auto the_result = at::bincount(self, weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 100: { // bincount
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bincount(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 101: { // bitwise_not
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bitwise_not(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 102: { // logical_not
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::logical_not(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 103: { // logical_xor
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::logical_xor(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 104: { // logical_and
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::logical_and(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 105: { // logical_or
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::logical_or(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 106: { // bmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::bmm(self, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 107: { // broadcast_tensors
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::broadcast_tensors(tensors);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 108: { // cat
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::cat(tensors, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 109: { // cat
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::cat(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 110: { // ceil
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::ceil(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 111: { // chain_matmul
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto matrices = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::chain_matmul(matrices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 112: { // chunk
          int64_t chunks = readAttribute<int64_t>("chunks");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::chunk(self, chunks, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 113: { // chunk
          int64_t chunks = readAttribute<int64_t>("chunks");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::chunk(self, chunks);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 114: { // clamp
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::clamp(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 115: { // clamp_max
          at::Scalar max = readScalarAttribute("max");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::clamp_max(self, max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 116: { // clamp_min
          at::Scalar min = readScalarAttribute("min");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::clamp_min(self, min);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 117: { // cudnn_is_acceptable
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cudnn_is_acceptable(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 118: { // constant_pad_nd
          auto pad = readIntArrayRef("pad");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::constant_pad_nd(self, pad, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 119: { // constant_pad_nd
          auto pad = readIntArrayRef("pad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::constant_pad_nd(self, pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 120: { // contiguous
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.contiguous();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 121: { // convolution
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 122: { // convolution_overrideable
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 123: { // convolution_backward_overrideable
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto input = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 124: { // _convolution
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 125: { // _convolution_nogroup
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 126: { // _convolution_double_backward
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool transposed = readAttribute<int64_t>("transposed");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto ggI = peek(0, 6);
              auto ggW = peek(1, 6);
              auto ggb = peek(2, 6);
              auto gO = peek(3, 6);
              auto weight = peek(4, 6);
              auto self = peek(5, 6);
              auto the_result = at::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 127: { // conv1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride, padding, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 128: { // conv1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 129: { // conv1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 130: { // conv1d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 131: { // conv1d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv1d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 132: { // conv1d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv1d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 133: { // conv2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride, padding, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 134: { // conv2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 135: { // conv2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 136: { // conv2d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 137: { // conv2d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv2d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 138: { // conv2d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv2d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 139: { // conv3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride, padding, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 140: { // conv3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 141: { // conv3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 142: { // conv3d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 143: { // conv3d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv3d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 144: { // conv3d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv3d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 145: { // conv_tbc
          int64_t pad = readAttribute<int64_t>("pad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_tbc(self, weight, bias, pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 146: { // conv_tbc
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_tbc(self, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 147: { // conv_tbc_backward
          int64_t pad = readAttribute<int64_t>("pad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 4);
              auto input = peek(1, 4);
              auto weight = peek(2, 4);
              auto bias = peek(3, 4);
              auto the_result = at::conv_tbc_backward(self, input, weight, bias, pad);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 148: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 149: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 150: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 151: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 152: { // conv_transpose1d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 153: { // conv_transpose1d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose1d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 154: { // conv_transpose1d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv_transpose1d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 155: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 156: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 157: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 158: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 159: { // conv_transpose2d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 160: { // conv_transpose2d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose2d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 161: { // conv_transpose2d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv_transpose2d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 162: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 163: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 164: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 165: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 166: { // conv_transpose3d
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 167: { // conv_transpose3d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::conv_transpose3d(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 168: { // conv_transpose3d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::conv_transpose3d(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 169: { // _copy_from
          bool non_blocking = readAttribute<int64_t>("non_blocking");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto dst = peek(1, 2);
              auto the_result = at::_copy_from(self, dst, non_blocking);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 170: { // _copy_from
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto dst = peek(1, 2);
              auto the_result = at::_copy_from(self, dst);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 171: { // cos
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cos(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 172: { // cosh
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cosh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 173: { // cosine_embedding_loss
          double margin = readAttribute<float>("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::cosine_embedding_loss(input1, input2, target, margin, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 174: { // cosine_embedding_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::cosine_embedding_loss(input1, input2, target, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 175: { // cosine_embedding_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::cosine_embedding_loss(input1, input2, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 176: { // cudnn_affine_grid_generator
          int64_t N = readAttribute<int64_t>("N");
          int64_t C = readAttribute<int64_t>("C");
          int64_t H = readAttribute<int64_t>("H");
          int64_t W = readAttribute<int64_t>("W");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto theta = peek(0, 1);
              auto the_result = at::cudnn_affine_grid_generator(theta, N, C, H, W);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 177: { // cudnn_affine_grid_generator_backward
          int64_t N = readAttribute<int64_t>("N");
          int64_t C = readAttribute<int64_t>("C");
          int64_t H = readAttribute<int64_t>("H");
          int64_t W = readAttribute<int64_t>("W");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 1);
              auto the_result = at::cudnn_affine_grid_generator_backward(grad, N, C, H, W);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 178: { // cudnn_batch_norm
          bool training = readAttribute<int64_t>("training");
          double exponential_average_factor = readAttribute<float>("exponential_average_factor");
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 179: { // cudnn_batch_norm_backward
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 8);
              auto grad_output = peek(1, 8);
              auto weight = peek(2, 8);
              auto running_mean = peek(3, 8);
              auto running_var = peek(4, 8);
              auto save_mean = peek(5, 8);
              auto save_var = peek(6, 8);
              auto reserveSpace = peek(7, 8);
              auto the_result = at::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 180: { // cudnn_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 181: { // cudnn_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 182: { // cudnn_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 183: { // cudnn_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<2>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 184: { // cudnn_convolution_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 185: { // cudnn_convolution_transpose
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 186: { // cudnn_convolution_transpose
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 187: { // cudnn_convolution_transpose_backward
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<2>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 188: { // cudnn_convolution_transpose_backward_input
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 189: { // cudnn_convolution_transpose_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 190: { // cudnn_grid_sampler
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::cudnn_grid_sampler(self, grid);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 191: { // cudnn_grid_sampler_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grid = peek(1, 3);
              auto grad_output = peek(2, 3);
              auto the_result = at::cudnn_grid_sampler_backward(self, grid, grad_output);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 192: { // cummax
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cummax(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 193: { // cummin
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cummin(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 194: { // cumprod
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cumprod(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 195: { // cumsum
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cumsum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 196: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 197: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 198: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 199: { // ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 200: { // ctc_loss
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 201: { // ctc_loss
          int64_t blank = readAttribute<int64_t>("blank");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 202: { // ctc_loss
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 203: { // ctc_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 4);
              auto targets = peek(1, 4);
              auto input_lengths = peek(2, 4);
              auto target_lengths = peek(3, 4);
              auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 204: { // _ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 205: { // _ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 206: { // _ctc_loss
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto log_probs = peek(0, 2);
              auto targets = peek(1, 2);
              auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 207: { // _ctc_loss_backward
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          bool zero_infinity = readAttribute<int64_t>("zero_infinity");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 5);
              auto log_probs = peek(1, 5);
              auto targets = peek(2, 5);
              auto neg_log_likelihood = peek(3, 5);
              auto log_alpha = peek(4, 5);
              auto the_result = at::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 208: { // _ctc_loss_backward
          auto input_lengths = readIntArrayRef("input_lengths");
          auto target_lengths = readIntArrayRef("target_lengths");
          int64_t blank = readAttribute<int64_t>("blank");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 5);
              auto log_probs = peek(1, 5);
              auto targets = peek(2, 5);
              auto neg_log_likelihood = peek(3, 5);
              auto log_alpha = peek(4, 5);
              auto the_result = at::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 209: { // det
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::det(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 210: { // diag_embed
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          int64_t dim2 = readAttribute<int64_t>("dim2");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self, offset, dim1, dim2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 211: { // diag_embed
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self, offset, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 212: { // diag_embed
          int64_t offset = readAttribute<int64_t>("offset");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self, offset);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 213: { // diag_embed
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diag_embed(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 214: { // diagflat
          int64_t offset = readAttribute<int64_t>("offset");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diagflat(self, offset);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 215: { // diagflat
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diagflat(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 216: { // diagonal
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          int64_t dim2 = readAttribute<int64_t>("dim2");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self, offset, dim1, dim2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 217: { // diagonal
          int64_t offset = readAttribute<int64_t>("offset");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self, offset, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 218: { // diagonal
          int64_t offset = readAttribute<int64_t>("offset");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self, offset);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 219: { // diagonal
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diagonal(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 220: { // div
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::div(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 221: { // div
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::div(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 222: { // dot
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto tensor = peek(1, 2);
              auto the_result = at::dot(self, tensor);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 223: { // embedding
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 224: { // embedding
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 225: { // embedding
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices, padding_idx);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 226: { // embedding
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding(weight, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 227: { // embedding_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 228: { // embedding_dense_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 229: { // embedding_sparse_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          int64_t padding_idx = readAttribute<int64_t>("padding_idx");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 230: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          bool include_last_offset = readAttribute<int64_t>("include_last_offset");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 4);
              auto indices = peek(1, 4);
              auto offsets = peek(2, 4);
              auto per_sample_weights = peek(3, 4);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 231: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 4);
              auto indices = peek(1, 4);
              auto offsets = peek(2, 4);
              auto per_sample_weights = peek(3, 4);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 232: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 233: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 234: { // embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 235: { // embedding_bag
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::embedding_bag(weight, indices, offsets);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 236: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          bool include_last_offset = readAttribute<int64_t>("include_last_offset");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 4);
              auto indices = peek(1, 4);
              auto offsets = peek(2, 4);
              auto per_sample_weights = peek(3, 4);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 237: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 4);
              auto indices = peek(1, 4);
              auto offsets = peek(2, 4);
              auto per_sample_weights = peek(3, 4);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 238: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 239: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 240: { // _embedding_bag
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 241: { // _embedding_bag
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto weight = peek(0, 3);
              auto indices = peek(1, 3);
              auto offsets = peek(2, 3);
              auto the_result = at::_embedding_bag(weight, indices, offsets);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 242: { // _embedding_bag_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          bool sparse = readAttribute<int64_t>("sparse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 7);
              auto indices = peek(1, 7);
              auto offsets = peek(2, 7);
              auto offset2bag = peek(3, 7);
              auto bag_size = peek(4, 7);
              auto maximum_indices = peek(5, 7);
              auto per_sample_weights = peek(6, 7);
              auto the_result = at::_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 243: { // _embedding_bag_sparse_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 6);
              auto indices = peek(1, 6);
              auto offsets = peek(2, 6);
              auto offset2bag = peek(3, 6);
              auto bag_size = peek(4, 6);
              auto per_sample_weights = peek(5, 6);
              auto the_result = at::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 244: { // _embedding_bag_dense_backward
          int64_t num_weights = readAttribute<int64_t>("num_weights");
          bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 7);
              auto indices = peek(1, 7);
              auto offsets = peek(2, 7);
              auto offset2bag = peek(3, 7);
              auto bag_size = peek(4, 7);
              auto maximum_indices = peek(5, 7);
              auto per_sample_weights = peek(6, 7);
              auto the_result = at::_embedding_bag_dense_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 245: { // _embedding_bag_per_sample_weights_backward
          int64_t mode = readAttribute<int64_t>("mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 5);
              auto weight = peek(1, 5);
              auto indices = peek(2, 5);
              auto offsets = peek(3, 5);
              auto offset2bag = peek(4, 5);
              auto the_result = at::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 246: { // erf
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::erf(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 247: { // erfc
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::erfc(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 248: { // exp
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::exp(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 249: { // expm1
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::expm1(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 250: { // expand
          auto size = readIntArrayRef("size");
          bool implicit = readAttribute<int64_t>("implicit");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.expand(size, implicit);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 251: { // expand
          auto size = readIntArrayRef("size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.expand(size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 252: { // expand_as
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.expand_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 253: { // flatten
          int64_t start_dim = readAttribute<int64_t>("start_dim");
          int64_t end_dim = readAttribute<int64_t>("end_dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::flatten(self, start_dim, end_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 254: { // flatten
          int64_t start_dim = readAttribute<int64_t>("start_dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::flatten(self, start_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 255: { // flatten
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::flatten(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 256: { // floor
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::floor(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 257: { // floor_divide
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::floor_divide(input, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 258: { // floor_divide
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::floor_divide(input, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 259: { // frac
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::frac(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 260: { // grid_sampler
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 261: { // grid_sampler_2d
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 262: { // grid_sampler_2d_backward
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto input = peek(1, 3);
              auto grid = peek(2, 3);
              auto the_result = at::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 263: { // grid_sampler_3d
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto grid = peek(1, 2);
              auto the_result = at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 264: { // grid_sampler_3d_backward
          int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
          int64_t padding_mode = readAttribute<int64_t>("padding_mode");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto input = peek(1, 3);
              auto grid = peek(2, 3);
              auto the_result = at::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 265: { // hinge_embedding_loss
          double margin = readAttribute<float>("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::hinge_embedding_loss(self, target, margin, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 266: { // hinge_embedding_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::hinge_embedding_loss(self, target, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 267: { // hinge_embedding_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::hinge_embedding_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 268: { // ger
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto vec2 = peek(1, 2);
              auto the_result = at::ger(self, vec2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 269: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 270: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::group_norm(input, num_groups, weight, bias, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 271: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::group_norm(input, num_groups, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 272: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::group_norm(input, num_groups, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 273: { // group_norm
          int64_t num_groups = readAttribute<int64_t>("num_groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::group_norm(input, num_groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 274: { // fft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::fft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 275: { // fft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::fft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 276: { // ifft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::ifft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 277: { // ifft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::ifft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 278: { // rfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rfft(self, signal_ndim, normalized, onesided);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 279: { // rfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rfft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 280: { // rfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rfft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 281: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          auto signal_sizes = readIntArrayRef("signal_sizes");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim, normalized, onesided, signal_sizes);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 282: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim, normalized, onesided);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 283: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool normalized = readAttribute<int64_t>("normalized");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim, normalized);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 284: { // irfft
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::irfft(self, signal_ndim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 285: { // _fft_with_size
          int64_t signal_ndim = readAttribute<int64_t>("signal_ndim");
          bool complex_input = readAttribute<int64_t>("complex_input");
          bool complex_output = readAttribute<int64_t>("complex_output");
          bool inverse = readAttribute<int64_t>("inverse");
          auto checked_signal_sizes = readIntArrayRef("checked_signal_sizes");
          bool normalized = readAttribute<int64_t>("normalized");
          bool onesided = readAttribute<int64_t>("onesided");
          auto output_sizes = readIntArrayRef("output_sizes");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_fft_with_size(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 286: { // _cufft_get_plan_cache_size
          int64_t device_index = readAttribute<int64_t>("device_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
      
              auto the_result = at::_cufft_get_plan_cache_size(device_index);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 287: { // _cufft_get_plan_cache_max_size
          int64_t device_index = readAttribute<int64_t>("device_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
      
              auto the_result = at::_cufft_get_plan_cache_max_size(device_index);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 288: { // index
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, InputSize());
              auto indices = peekSlice(1, InputSize() - 1, InputSize());
              auto the_result = at::index(self, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 289: { // index_copy
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto source = peek(2, 3);
              auto the_result = at::index_copy(self, dim, index, source);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 290: { // index_put
          bool accumulate = readAttribute<int64_t>("accumulate");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, InputSize());
              auto indices = peekSlice(1, InputSize() - 2, InputSize());
              auto values = peek(1, 2);
              auto the_result = at::index_put(self, indices, values, accumulate);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 291: { // index_put
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, InputSize());
              auto indices = peekSlice(1, InputSize() - 2, InputSize());
              auto values = peek(1, 2);
              auto the_result = at::index_put(self, indices, values);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 292: { // instance_norm
          bool use_input_stats = readAttribute<int64_t>("use_input_stats");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 293: { // inverse
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::inverse(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 294: { // _inverse_helper
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_inverse_helper(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 295: { // isclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          bool equal_nan = readAttribute<int64_t>("equal_nan");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other, rtol, atol, equal_nan);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 296: { // isclose
          double rtol = readAttribute<float>("rtol");
          double atol = readAttribute<float>("atol");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other, rtol, atol);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 297: { // isclose
          double rtol = readAttribute<float>("rtol");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other, rtol);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 298: { // isclose
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::isclose(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 299: { // isnan
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::isnan(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 300: { // is_distributed
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::is_distributed(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 301: { // is_floating_point
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::is_floating_point(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 302: { // is_complex
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::is_complex(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 303: { // is_nonzero
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::is_nonzero(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 304: { // is_same_size
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::is_same_size(self, other);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 305: { // is_signed
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::is_signed(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 306: { // kl_div
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::kl_div(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 307: { // kl_div
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::kl_div(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 308: { // kl_div_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::kl_div_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 309: { // kl_div_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::kl_div_backward(grad_output, self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 310: { // kthvalue
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::kthvalue(self, k, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 311: { // kthvalue
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::kthvalue(self, k, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 312: { // kthvalue
          int64_t k = readAttribute<int64_t>("k");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::kthvalue(self, k);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 313: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          double eps = readAttribute<float>("eps");
          bool cudnn_enable = readAttribute<int64_t>("cudnn_enable");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 314: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::layer_norm(input, normalized_shape, weight, bias, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 315: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::layer_norm(input, normalized_shape, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 316: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::layer_norm(input, normalized_shape, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 317: { // layer_norm
          auto normalized_shape = readIntArrayRef("normalized_shape");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::layer_norm(input, normalized_shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 318: { // native_layer_norm
          int64_t M = readAttribute<int64_t>("M");
          int64_t N = readAttribute<int64_t>("N");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::native_layer_norm(input, weight, bias, M, N, eps);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 319: { // native_layer_norm_backward
          int64_t M = readAttribute<int64_t>("M");
          int64_t N = readAttribute<int64_t>("N");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_out = peek(0, 5);
              auto input = peek(1, 5);
              auto mean = peek(2, 5);
              auto rstd = peek(3, 5);
              auto weight = peek(4, 5);
              auto the_result = at::native_layer_norm_backward(grad_out, input, mean, rstd, weight, M, N, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 320: { // linear
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::linear(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 321: { // linear
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::linear(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 322: { // mkldnn_linear
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::mkldnn_linear(input, weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 323: { // mkldnn_linear
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::mkldnn_linear(input, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 324: { // fbgemm_linear_int8_weight_fp32_activation
          at::Scalar weight_scale = readScalarAttribute("weight_scale");
          at::Scalar weight_zero_point = readScalarAttribute("weight_zero_point");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto packed = peek(2, 5);
              auto col_offsets = peek(3, 5);
              auto bias = peek(4, 5);
              auto the_result = at::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 325: { // fbgemm_linear_int8_weight
          at::Scalar weight_scale = readScalarAttribute("weight_scale");
          at::Scalar weight_zero_point = readScalarAttribute("weight_zero_point");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto packed = peek(2, 5);
              auto col_offsets = peek(3, 5);
              auto bias = peek(4, 5);
              auto the_result = at::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 326: { // fbgemm_pack_gemm_matrix_fp16
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::fbgemm_pack_gemm_matrix_fp16(input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 327: { // fbgemm_linear_fp16_weight_fp32_activation
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto packed_weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 328: { // fbgemm_linear_fp16_weight
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto packed_weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::fbgemm_linear_fp16_weight(input, packed_weight, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 329: { // fbgemm_pack_quantized_matrix
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::fbgemm_pack_quantized_matrix(input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 330: { // fbgemm_pack_quantized_matrix
          int64_t K = readAttribute<int64_t>("K");
          int64_t N = readAttribute<int64_t>("N");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::fbgemm_pack_quantized_matrix(input, K, N);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 331: { // log
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 332: { // log10
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log10(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 333: { // log1p
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log1p(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 334: { // log2
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log2(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 335: { // logdet
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::logdet(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 336: { // log_softmax
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log_softmax(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 337: { // _log_softmax
          int64_t dim = readAttribute<int64_t>("dim");
          bool half_to_float = readAttribute<int64_t>("half_to_float");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_log_softmax(self, dim, half_to_float);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 338: { // _log_softmax_backward_data
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto output = peek(1, 3);
              auto self = peek(2, 3);
              auto the_result = at::_log_softmax_backward_data(grad_output, output, dim, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 339: { // logsumexp
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::logsumexp(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 340: { // logsumexp
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::logsumexp(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 341: { // margin_ranking_loss
          double margin = readAttribute<float>("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::margin_ranking_loss(input1, input2, target, margin, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 342: { // margin_ranking_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::margin_ranking_loss(input1, input2, target, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 343: { // margin_ranking_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input1 = peek(0, 3);
              auto input2 = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::margin_ranking_loss(input1, input2, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 344: { // matmul
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::matmul(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 345: { // matrix_rank
          double tol = readAttribute<float>("tol");
          bool symmetric = readAttribute<int64_t>("symmetric");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self, tol, symmetric);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 346: { // matrix_rank
          double tol = readAttribute<float>("tol");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self, tol);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 347: { // matrix_rank
          bool symmetric = readAttribute<int64_t>("symmetric");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self, symmetric);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 348: { // matrix_rank
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::matrix_rank(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 349: { // matrix_power
          int64_t n = readAttribute<int64_t>("n");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::matrix_power(self, n);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 350: { // max
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 351: { // max
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 352: { // max_values
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_values(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 353: { // max_values
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_values(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 354: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 355: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 356: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 357: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 358: { // max_pool1d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d_with_indices(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 359: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 360: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 361: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 362: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 363: { // max_pool1d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool1d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 364: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 365: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 366: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 367: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 368: { // max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 369: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 370: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 371: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 372: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 373: { // mkldnn_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_max_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 374: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 375: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 376: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 377: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 378: { // quantized_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::quantized_max_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 379: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 380: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 381: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 382: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 383: { // max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 384: { // mean
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mean(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 385: { // mean
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mean(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 386: { // mean
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mean(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 387: { // median
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::median(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 388: { // median
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::median(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 389: { // min
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::min(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 390: { // min
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::min(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 391: { // min_values
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::min_values(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 392: { // min_values
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::min_values(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 393: { // mkldnn_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 394: { // mkldnn_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool bias_defined = readAttribute<int64_t>("bias_defined");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::mkldnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 395: { // mkldnn_convolution_backward_weights
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool bias_defined = readAttribute<int64_t>("bias_defined");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 396: { // mkldnn_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::mkldnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 397: { // miopen_batch_norm
          bool training = readAttribute<int64_t>("training");
          double exponential_average_factor = readAttribute<float>("exponential_average_factor");
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 398: { // miopen_batch_norm_backward
          double epsilon = readAttribute<float>("epsilon");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 7);
              auto grad_output = peek(1, 7);
              auto weight = peek(2, 7);
              auto running_mean = peek(3, 7);
              auto running_var = peek(4, 7);
              auto save_mean = peek(5, 7);
              auto save_var = peek(6, 7);
              auto the_result = at::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 399: { // miopen_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 400: { // miopen_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::miopen_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 401: { // miopen_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::miopen_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 402: { // miopen_convolution_backward_bias
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::miopen_convolution_backward_bias(grad_output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 403: { // miopen_convolution_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::miopen_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 404: { // miopen_convolution_transpose
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 405: { // miopen_convolution_transpose_backward
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::miopen_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 406: { // miopen_convolution_transpose_backward_input
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 407: { // miopen_convolution_transpose_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::miopen_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 408: { // miopen_depthwise_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 409: { // miopen_depthwise_convolution_backward_input
          auto self_size = readIntArrayRef("self_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::miopen_depthwise_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 410: { // miopen_depthwise_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::miopen_depthwise_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 411: { // miopen_depthwise_convolution_backward_weight
          auto weight_size = readIntArrayRef("weight_size");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          bool benchmark = readAttribute<int64_t>("benchmark");
          bool deterministic = readAttribute<int64_t>("deterministic");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::miopen_depthwise_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 412: { // miopen_rnn
          int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
          int64_t mode = readAttribute<int64_t>("mode");
          int64_t hidden_size = readAttribute<int64_t>("hidden_size");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          bool batch_first = readAttribute<int64_t>("batch_first");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          auto batch_sizes = readIntArrayRef("batch_sizes");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto weight = peekSlice(1, InputSize() - 4, InputSize());
              auto hx = peek(1, 4);
              auto cx = peek(2, 4);
              auto dropout_state = peek(3, 4);
              auto the_result = at::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 413: { // mm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::mm(self, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 414: { // _sparse_mm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto sparse = peek(0, 2);
              auto dense = peek(1, 2);
              auto the_result = at::_sparse_mm(sparse, dense);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 415: { // mode
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mode(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 416: { // mode
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mode(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 417: { // mode
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mode(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 418: { // mul
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::mul(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 419: { // mul
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mul(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 420: { // mv
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto vec = peek(1, 2);
              auto the_result = at::mv(self, vec);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 421: { // mvlgamma
          int64_t p = readAttribute<int64_t>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mvlgamma(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 422: { // narrow_copy
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t length = readAttribute<int64_t>("length");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.narrow_copy(dim, start, length);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 423: { // narrow
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t length = readAttribute<int64_t>("length");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::narrow(self, dim, start, length);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 424: { // native_batch_norm
          bool training = readAttribute<int64_t>("training");
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 425: { // batch_norm_stats
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 1);
              auto the_result = at::batch_norm_stats(input, eps);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 426: { // batch_norm_elemt
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto weight = peek(1, 5);
              auto bias = peek(2, 5);
              auto mean = peek(3, 5);
              auto invstd = peek(4, 5);
              auto the_result = at::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 427: { // batch_norm_gather_stats
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          int64_t count = readAttribute<int64_t>("count");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto mean = peek(1, 5);
              auto invstd = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 428: { // batch_norm_gather_stats_with_counts
          double momentum = readAttribute<float>("momentum");
          double eps = readAttribute<float>("eps");
          auto counts = readIntArrayRef("counts");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto mean = peek(1, 5);
              auto invstd = peek(2, 5);
              auto running_mean = peek(3, 5);
              auto running_var = peek(4, 5);
              auto the_result = at::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 429: { // native_batch_norm_backward
          bool train = readAttribute<int64_t>("train");
          double eps = readAttribute<float>("eps");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_out = peek(0, 7);
              auto input = peek(1, 7);
              auto weight = peek(2, 7);
              auto running_mean = peek(3, 7);
              auto running_var = peek(4, 7);
              auto save_mean = peek(5, 7);
              auto save_invstd = peek(6, 7);
              auto the_result = at::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 430: { // batch_norm_backward_reduce
          bool input_g = readAttribute<int64_t>("input_g");
          bool weight_g = readAttribute<int64_t>("weight_g");
          bool bias_g = readAttribute<int64_t>("bias_g");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_out = peek(0, 5);
              auto input = peek(1, 5);
              auto mean = peek(2, 5);
              auto invstd = peek(3, 5);
              auto weight = peek(4, 5);
              auto the_result = at::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
              return true;
          };
      } break;
      case 431: { // batch_norm_backward_elemt
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_out = peek(0, 7);
              auto input = peek(1, 7);
              auto mean = peek(2, 7);
              auto invstd = peek(3, 7);
              auto weight = peek(4, 7);
              auto mean_dy = peek(5, 7);
              auto mean_dy_xmu = peek(6, 7);
              auto the_result = at::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 432: { // batch_norm_update_stats
          double momentum = readAttribute<float>("momentum");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto running_mean = peek(1, 3);
              auto running_var = peek(2, 3);
              auto the_result = at::batch_norm_update_stats(input, running_mean, running_var, momentum);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 433: { // _nnpack_available
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
      
              auto the_result = at::_nnpack_available();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 434: { // _nnpack_spatial_convolution
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 435: { // _nnpack_spatial_convolution
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution(input, weight, bias, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 436: { // _nnpack_spatial_convolution_backward
          auto padding = readIntArrayRef("padding");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution_backward(input, grad_output, weight, padding, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 437: { // _nnpack_spatial_convolution_backward_input
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 3);
              auto grad_output = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::_nnpack_spatial_convolution_backward_input(input, grad_output, weight, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 438: { // _nnpack_spatial_convolution_backward_weight
          auto weightsize = readIntArrayRef("weightsize");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto grad_output = peek(1, 2);
              auto the_result = at::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 439: { // pairwise_distance
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2, p, eps, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 440: { // pairwise_distance
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2, p, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 441: { // pairwise_distance
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 442: { // pairwise_distance
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::pairwise_distance(x1, x2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 443: { // cdist
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cdist(x1, x2, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 444: { // cdist
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cdist(x1, x2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 445: { // _cdist_backward
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 4);
              auto x1 = peek(1, 4);
              auto x2 = peek(2, 4);
              auto cdist = peek(3, 4);
              auto the_result = at::_cdist_backward(grad, x1, x2, p, cdist);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 446: { // pdist
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::pdist(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 447: { // pdist
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::pdist(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 448: { // _pdist_forward
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_pdist_forward(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 449: { // _pdist_forward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_pdist_forward(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 450: { // _pdist_backward
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 3);
              auto self = peek(1, 3);
              auto pdist = peek(2, 3);
              auto the_result = at::_pdist_backward(grad, self, p, pdist);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 451: { // cosine_similarity
          int64_t dim = readAttribute<int64_t>("dim");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cosine_similarity(x1, x2, dim, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 452: { // cosine_similarity
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cosine_similarity(x1, x2, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 453: { // cosine_similarity
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x1 = peek(0, 2);
              auto x2 = peek(1, 2);
              auto the_result = at::cosine_similarity(x1, x2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 454: { // permute
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.permute(dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 455: { // numpy_T
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.numpy_T();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 456: { // pixel_shuffle
          int64_t upscale_factor = readAttribute<int64_t>("upscale_factor");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::pixel_shuffle(self, upscale_factor);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 457: { // is_pinned
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.is_pinned();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 458: { // pin_memory
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.pin_memory();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 459: { // pinverse
          double rcond = readAttribute<float>("rcond");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::pinverse(self, rcond);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 460: { // pinverse
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::pinverse(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 461: { // poisson_nll_loss
          bool log_input = readAttribute<int64_t>("log_input");
          bool full = readAttribute<int64_t>("full");
          double eps = readAttribute<float>("eps");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::poisson_nll_loss(input, target, log_input, full, eps, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 462: { // reciprocal
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::reciprocal(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 463: { // neg
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::neg(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 464: { // repeat
          auto repeats = readIntArrayRef("repeats");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.repeat(repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 465: { // repeat_interleave
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto repeats = peek(0, 1);
              auto the_result = at::repeat_interleave(repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 466: { // repeat_interleave
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto repeats = peek(1, 2);
              auto the_result = at::repeat_interleave(self, repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 467: { // repeat_interleave
          int64_t repeats = readAttribute<int64_t>("repeats");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::repeat_interleave(self, repeats);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 468: { // reshape
          auto shape = readIntArrayRef("shape");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::reshape(self, shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 469: { // _mkldnn_reshape
          auto shape = readIntArrayRef("shape");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_mkldnn_reshape(self, shape);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 470: { // reshape_as
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.reshape_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 471: { // round
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::round(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 472: { // rrelu
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          bool training = readAttribute<int64_t>("training");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self, lower, upper, training);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 473: { // rrelu
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self, lower, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 474: { // rrelu
          at::Scalar lower = readScalarAttribute("lower");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self, lower);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 475: { // rrelu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rrelu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 476: { // relu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::relu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 477: { // prelu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::prelu(self, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 478: { // prelu_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::prelu_backward(grad_output, self, weight);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 479: { // gelu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::gelu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 480: { // gelu_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::gelu_backward(grad, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 481: { // hardshrink
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::hardshrink(self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 482: { // hardshrink
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::hardshrink(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 483: { // hardshrink_backward
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_out = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::hardshrink_backward(grad_out, self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 484: { // rsqrt
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rsqrt(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 485: { // select
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t index = readAttribute<int64_t>("index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::select(self, dim, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 486: { // selu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::selu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 487: { // celu
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::celu(self, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 488: { // celu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::celu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 489: { // sigmoid
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sigmoid(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 490: { // sin
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sin(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 491: { // sinh
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sinh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 492: { // detach
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::detach(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 493: { // size
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::size(self, dim);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 494: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t end = readAttribute<int64_t>("end");
          int64_t step = readAttribute<int64_t>("step");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim, start, end, step);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 495: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          int64_t end = readAttribute<int64_t>("end");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim, start, end);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 496: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          int64_t start = readAttribute<int64_t>("start");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim, start);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 497: { // slice
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::slice(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 498: { // slice
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::slice(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 499: { // slogdet
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::slogdet(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 500: { // smm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::smm(self, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 501: { // softmax
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::softmax(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 502: { // _softmax
          int64_t dim = readAttribute<int64_t>("dim");
          bool half_to_float = readAttribute<int64_t>("half_to_float");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_softmax(self, dim, half_to_float);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 503: { // _softmax_backward_data
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto output = peek(1, 3);
              auto self = peek(2, 3);
              auto the_result = at::_softmax_backward_data(grad_output, output, dim, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 504: { // split
          int64_t split_size = readAttribute<int64_t>("split_size");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::split(self, split_size, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 505: { // split
          int64_t split_size = readAttribute<int64_t>("split_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::split(self, split_size);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 506: { // split_with_sizes
          auto split_sizes = readIntArrayRef("split_sizes");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::split_with_sizes(self, split_sizes, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 507: { // split_with_sizes
          auto split_sizes = readIntArrayRef("split_sizes");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::split_with_sizes(self, split_sizes);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 508: { // squeeze
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::squeeze(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 509: { // squeeze
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::squeeze(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 510: { // sspaddmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::sspaddmm(self, mat1, mat2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 511: { // sspaddmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::sspaddmm(self, mat1, mat2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 512: { // sspaddmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::sspaddmm(self, mat1, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 513: { // stack
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::stack(tensors, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 514: { // stack
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::stack(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 515: { // stft
          int64_t n_fft = readAttribute<int64_t>("n_fft");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::stft(self, n_fft);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 516: { // stride
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::stride(self, dim);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 517: { // sum
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sum(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 518: { // sum
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sum(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 519: { // sum
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 520: { // sum_to_size
          auto size = readIntArrayRef("size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.sum_to_size(size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 521: { // sqrt
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sqrt(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 522: { // square
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::square(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 523: { // std
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 524: { // std
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 525: { // std
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 526: { // std
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 527: { // std
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 528: { // std_mean
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 529: { // std_mean
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 530: { // std_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 531: { // std_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 532: { // std_mean
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::std_mean(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 533: { // prod
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::prod(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 534: { // prod
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::prod(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 535: { // prod
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::prod(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 536: { // t
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::t(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 537: { // tan
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::tan(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 538: { // tanh
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::tanh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 539: { // tensordot
          auto dims_self = readIntArrayRef("dims_self");
          auto dims_other = readIntArrayRef("dims_other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::tensordot(self, other, dims_self, dims_other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 540: { // threshold
          at::Scalar threshold = readScalarAttribute("threshold");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::threshold(self, threshold, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 541: { // threshold_backward
          at::Scalar threshold = readScalarAttribute("threshold");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::threshold_backward(grad_output, self, threshold);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 542: { // transpose
          int64_t dim0 = readAttribute<int64_t>("dim0");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::transpose(self, dim0, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 543: { // _mkldnn_transpose
          int64_t dim0 = readAttribute<int64_t>("dim0");
          int64_t dim1 = readAttribute<int64_t>("dim1");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_mkldnn_transpose(self, dim0, dim1);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 544: { // one_hot
          int64_t num_classes = readAttribute<int64_t>("num_classes");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::one_hot(self, num_classes);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 545: { // one_hot
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::one_hot(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 546: { // flip
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::flip(self, dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 547: { // roll
          auto shifts = readIntArrayRef("shifts");
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::roll(self, shifts, dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 548: { // roll
          auto shifts = readIntArrayRef("shifts");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::roll(self, shifts);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 549: { // rot90
          int64_t k = readAttribute<int64_t>("k");
          auto dims = readIntArrayRef("dims");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rot90(self, k, dims);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 550: { // rot90
          int64_t k = readAttribute<int64_t>("k");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rot90(self, k);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 551: { // rot90
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rot90(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 552: { // trapz
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto y = peek(0, 2);
              auto x = peek(1, 2);
              auto the_result = at::trapz(y, x, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 553: { // trapz
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto y = peek(0, 2);
              auto x = peek(1, 2);
              auto the_result = at::trapz(y, x);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 554: { // trapz
          double dx = readAttribute<float>("dx");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto y = peek(0, 1);
              auto the_result = at::trapz(y, dx, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 555: { // trapz
          double dx = readAttribute<float>("dx");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto y = peek(0, 1);
              auto the_result = at::trapz(y, dx);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 556: { // trapz
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto y = peek(0, 1);
              auto the_result = at::trapz(y);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 557: { // _trilinear
          auto expand1 = readIntArrayRef("expand1");
          auto expand2 = readIntArrayRef("expand2");
          auto expand3 = readIntArrayRef("expand3");
          auto sumdim = readIntArrayRef("sumdim");
          int64_t unroll_dim = readAttribute<int64_t>("unroll_dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto i1 = peek(0, 3);
              auto i2 = peek(1, 3);
              auto i3 = peek(2, 3);
              auto the_result = at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 558: { // _trilinear
          auto expand1 = readIntArrayRef("expand1");
          auto expand2 = readIntArrayRef("expand2");
          auto expand3 = readIntArrayRef("expand3");
          auto sumdim = readIntArrayRef("sumdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto i1 = peek(0, 3);
              auto i2 = peek(1, 3);
              auto i3 = peek(2, 3);
              auto the_result = at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 559: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          bool swap = readAttribute<int64_t>("swap");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 560: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          bool swap = readAttribute<int64_t>("swap");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 561: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          double eps = readAttribute<float>("eps");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 562: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          double p = readAttribute<float>("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 563: { // triplet_margin_loss
          double margin = readAttribute<float>("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 564: { // triplet_margin_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto anchor = peek(0, 3);
              auto positive = peek(1, 3);
              auto negative = peek(2, 3);
              auto the_result = at::triplet_margin_loss(anchor, positive, negative);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 565: { // trunc
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::trunc(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 566: { // type_as
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.type_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 567: { // _has_compatible_shallow_copy_type
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto from = peek(1, 2);
              auto the_result = at::_has_compatible_shallow_copy_type(self, from);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 568: { // _unique
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique(self, sorted, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 569: { // _unique
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique(self, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 570: { // _unique
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 571: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim, sorted, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 572: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim, sorted, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 573: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 574: { // unique_dim
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 575: { // unique_consecutive
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_consecutive(self, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 576: { // unique_consecutive
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_consecutive(self, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 577: { // unique_consecutive
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_consecutive(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 578: { // unique_dim_consecutive
          int64_t dim = readAttribute<int64_t>("dim");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim_consecutive(self, dim, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 579: { // unique_dim_consecutive
          int64_t dim = readAttribute<int64_t>("dim");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim_consecutive(self, dim, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 580: { // unique_dim_consecutive
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unique_dim_consecutive(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 581: { // _unique2
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          bool return_counts = readAttribute<int64_t>("return_counts");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self, sorted, return_inverse, return_counts);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 582: { // _unique2
          bool sorted = readAttribute<int64_t>("sorted");
          bool return_inverse = readAttribute<int64_t>("return_inverse");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self, sorted, return_inverse);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 583: { // _unique2
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 584: { // _unique2
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unique2(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 585: { // _unsafe_view
          auto size = readIntArrayRef("size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_unsafe_view(self, size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 586: { // unsqueeze
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unsqueeze(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 587: { // var
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 588: { // var
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 589: { // var
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 590: { // var
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 591: { // var
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 592: { // var_mean
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 593: { // var_mean
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 594: { // var_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, dim, unbiased, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 595: { // var_mean
          auto dim = readIntArrayRef("dim");
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, dim, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 596: { // var_mean
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::var_mean(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 597: { // view_as
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = self.view_as(other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 598: { // where
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto condition = peek(0, 3);
              auto self = peek(1, 3);
              auto other = peek(2, 3);
              auto the_result = at::where(condition, self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 599: { // where
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto condition = peek(0, 1);
              auto the_result = at::where(condition);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 600: { // _s_where
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto condition = peek(0, 3);
              auto self = peek(1, 3);
              auto other = peek(2, 3);
              auto the_result = at::_s_where(condition, self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 601: { // norm_except_dim
          int64_t pow = readAttribute<int64_t>("pow");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 1);
              auto the_result = at::norm_except_dim(v, pow, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 602: { // norm_except_dim
          int64_t pow = readAttribute<int64_t>("pow");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 1);
              auto the_result = at::norm_except_dim(v, pow);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 603: { // norm_except_dim
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 1);
              auto the_result = at::norm_except_dim(v);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 604: { // _weight_norm
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm(v, g, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 605: { // _weight_norm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm(v, g);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 606: { // _weight_norm_cuda_interface
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm_cuda_interface(v, g, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 607: { // _weight_norm_cuda_interface
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto v = peek(0, 2);
              auto g = peek(1, 2);
              auto the_result = at::_weight_norm_cuda_interface(v, g);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 608: { // _weight_norm_cuda_interface_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_w = peek(0, 4);
              auto saved_v = peek(1, 4);
              auto saved_g = peek(2, 4);
              auto saved_norms = peek(3, 4);
              auto the_result = at::_weight_norm_cuda_interface_backward(grad_w, saved_v, saved_g, saved_norms, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 609: { // _weight_norm_differentiable_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_w = peek(0, 4);
              auto saved_v = peek(1, 4);
              auto saved_g = peek(2, 4);
              auto saved_norms = peek(3, 4);
              auto the_result = at::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 610: { // _standard_gamma_grad
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::_standard_gamma_grad(self, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 611: { // _standard_gamma
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_standard_gamma(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 612: { // _dirichlet_grad
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto x = peek(0, 3);
              auto alpha = peek(1, 3);
              auto total = peek(2, 3);
              auto the_result = at::_dirichlet_grad(x, alpha, total);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 613: { // _sample_dirichlet
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_sample_dirichlet(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 614: { // poisson
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::poisson(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 615: { // native_norm
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::native_norm(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 616: { // native_norm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::native_norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 617: { // _sparse_sum
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_sparse_sum(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 618: { // _sparse_sum
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_sparse_sum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 619: { // _sparse_sum_backward
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::_sparse_sum_backward(grad, self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 620: { // norm
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::norm(self, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 621: { // norm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 622: { // frobenius_norm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::frobenius_norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 623: { // frobenius_norm
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::frobenius_norm(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 624: { // frobenius_norm
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::frobenius_norm(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 625: { // nuclear_norm
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 626: { // nuclear_norm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 627: { // nuclear_norm
          auto dim = readIntArrayRef("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 628: { // nuclear_norm
          auto dim = readIntArrayRef("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::nuclear_norm(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 629: { // clone
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::clone(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 630: { // pow
          at::Scalar exponent = readScalarAttribute("exponent");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::pow(self, exponent);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 631: { // sub
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::sub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 632: { // sub
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::sub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 633: { // sub
          at::Scalar other = readScalarAttribute("other");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 634: { // sub
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 635: { // rsub
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::rsub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 636: { // rsub
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::rsub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 637: { // rsub
          at::Scalar other = readScalarAttribute("other");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rsub(self, other, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 638: { // rsub
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::rsub(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 639: { // _sparse_addmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto sparse = peek(1, 3);
              auto dense = peek(2, 3);
              auto the_result = at::_sparse_addmm(self, sparse, dense, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 640: { // _sparse_addmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto sparse = peek(1, 3);
              auto dense = peek(2, 3);
              auto the_result = at::_sparse_addmm(self, sparse, dense, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 641: { // _sparse_addmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto sparse = peek(1, 3);
              auto dense = peek(2, 3);
              auto the_result = at::_sparse_addmm(self, sparse, dense);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 642: { // addmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::addmm(self, mat1, mat2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 643: { // addmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::addmm(self, mat1, mat2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 644: { // addmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mat1 = peek(1, 3);
              auto mat2 = peek(2, 3);
              auto the_result = at::addmm(self, mat1, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 645: { // sparse_mask
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = self.sparse_mask(mask);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 646: { // to_dense
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.to_dense();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 647: { // to_dense_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto input = peek(1, 2);
              auto the_result = at::to_dense_backward(grad, input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 648: { // sparse_dim
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.sparse_dim();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 649: { // _dimI
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self._dimI();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 650: { // dense_dim
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.dense_dim();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 651: { // _dimV
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self._dimV();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 652: { // _nnz
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self._nnz();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 653: { // coalesce
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.coalesce();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 654: { // is_coalesced
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.is_coalesced();
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 655: { // _indices
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self._indices();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 656: { // _values
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self._values();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 657: { // indices
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.indices();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 658: { // values
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.values();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 659: { // hspmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto mat1 = peek(0, 2);
              auto mat2 = peek(1, 2);
              auto the_result = at::hspmm(mat1, mat2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 660: { // unbind
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unbind(self, dim);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 661: { // unbind
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::unbind(self);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 662: { // to_sparse
          int64_t sparse_dim = readAttribute<int64_t>("sparse_dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.to_sparse(sparse_dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 663: { // to_sparse
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.to_sparse();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 664: { // to_mkldnn
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.to_mkldnn();
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 665: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          int64_t groups = readAttribute<int64_t>("groups");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 666: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 667: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 668: { // mkldnn_reorder_conv2d_weight
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 669: { // mkldnn_reorder_conv2d_weight
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_reorder_conv2d_weight(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 670: { // to_mkldnn_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto input = peek(1, 2);
              auto the_result = at::to_mkldnn_backward(grad, input);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 671: { // dequantize
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::dequantize(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 672: { // q_zero_point
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::q_zero_point(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 673: { // q_per_channel_scales
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::q_per_channel_scales(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 674: { // q_per_channel_zero_points
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::q_per_channel_zero_points(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 675: { // q_per_channel_axis
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::q_per_channel_axis(self);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 676: { // int_repr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::int_repr(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 677: { // _make_per_tensor_quantized_tensor
          double scale = readAttribute<float>("scale");
          int64_t zero_point = readAttribute<int64_t>("zero_point");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_make_per_tensor_quantized_tensor(self, scale, zero_point);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 678: { // _make_per_channel_quantized_tensor
          int64_t axis = readAttribute<int64_t>("axis");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto scale = peek(1, 3);
              auto zero_point = peek(2, 3);
              auto the_result = at::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 679: { // fake_quantize_per_tensor_affine
          double scale = readAttribute<float>("scale");
          int64_t zero_point = readAttribute<int64_t>("zero_point");
          int64_t quant_min = readAttribute<int64_t>("quant_min");
          int64_t quant_max = readAttribute<int64_t>("quant_max");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 680: { // fake_quantize_per_tensor_affine_backward
          double scale = readAttribute<float>("scale");
          int64_t zero_point = readAttribute<int64_t>("zero_point");
          int64_t quant_min = readAttribute<int64_t>("quant_min");
          int64_t quant_max = readAttribute<int64_t>("quant_max");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::fake_quantize_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 681: { // fake_quantize_per_channel_affine
          int64_t axis = readAttribute<int64_t>("axis");
          int64_t quant_min = readAttribute<int64_t>("quant_min");
          int64_t quant_max = readAttribute<int64_t>("quant_max");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto scale = peek(1, 3);
              auto zero_point = peek(2, 3);
              auto the_result = at::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 682: { // fake_quantize_per_channel_affine_backward
          int64_t axis = readAttribute<int64_t>("axis");
          int64_t quant_min = readAttribute<int64_t>("quant_min");
          int64_t quant_max = readAttribute<int64_t>("quant_max");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 4);
              auto self = peek(1, 4);
              auto scale = peek(2, 4);
              auto zero_point = peek(3, 4);
              auto the_result = at::fake_quantize_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 683: { // meshgrid
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::meshgrid(tensors);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 684: { // cartesian_prod
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::cartesian_prod(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 685: { // combinations
          int64_t r = readAttribute<int64_t>("r");
          bool with_replacement = readAttribute<int64_t>("with_replacement");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::combinations(self, r, with_replacement);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 686: { // combinations
          int64_t r = readAttribute<int64_t>("r");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::combinations(self, r);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 687: { // combinations
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::combinations(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 688: { // item
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.item();
                if(OutputSize() > 0) {assignTo(Output(0),self.scalar_type(), the_result);}
              return true;
          };
      } break;
      case 689: { // _local_scalar_dense
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_local_scalar_dense(self);
                if(OutputSize() > 0) {assignTo(Output(0),self.scalar_type(), the_result);}
              return true;
          };
      } break;
      case 690: { // _thnn_fused_lstm_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input_gates = peek(0, 5);
              auto hidden_gates = peek(1, 5);
              auto cx = peek(2, 5);
              auto input_bias = peek(3, 5);
              auto hidden_bias = peek(4, 5);
              auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 691: { // _thnn_fused_lstm_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input_gates = peek(0, 4);
              auto hidden_gates = peek(1, 4);
              auto cx = peek(2, 4);
              auto input_bias = peek(3, 4);
              auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 692: { // _thnn_fused_lstm_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input_gates = peek(0, 3);
              auto hidden_gates = peek(1, 3);
              auto cx = peek(2, 3);
              auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 693: { // _thnn_fused_lstm_cell_backward
          bool has_bias = readAttribute<int64_t>("has_bias");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_hy = peek(0, 5);
              auto grad_cy = peek(1, 5);
              auto cx = peek(2, 5);
              auto cy = peek(3, 5);
              auto workspace = peek(4, 5);
              auto the_result = at::_thnn_fused_lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 694: { // _thnn_differentiable_lstm_cell_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_hy = peek(0, 8);
              auto grad_cy = peek(1, 8);
              auto input_gates = peek(2, 8);
              auto hidden_gates = peek(3, 8);
              auto input_bias = peek(4, 8);
              auto hidden_bias = peek(5, 8);
              auto cx = peek(6, 8);
              auto cy = peek(7, 8);
              auto the_result = at::_thnn_differentiable_lstm_cell_backward(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 695: { // _thnn_fused_gru_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input_gates = peek(0, 5);
              auto hidden_gates = peek(1, 5);
              auto hx = peek(2, 5);
              auto input_bias = peek(3, 5);
              auto hidden_bias = peek(4, 5);
              auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 696: { // _thnn_fused_gru_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input_gates = peek(0, 4);
              auto hidden_gates = peek(1, 4);
              auto hx = peek(2, 4);
              auto input_bias = peek(3, 4);
              auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 697: { // _thnn_fused_gru_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input_gates = peek(0, 3);
              auto hidden_gates = peek(1, 3);
              auto hx = peek(2, 3);
              auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 698: { // _thnn_fused_gru_cell_backward
          bool has_bias = readAttribute<int64_t>("has_bias");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_hy = peek(0, 2);
              auto workspace = peek(1, 2);
              auto the_result = at::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 699: { // _thnn_differentiable_gru_cell_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_hy = peek(0, 6);
              auto input_gates = peek(1, 6);
              auto hidden_gates = peek(2, 6);
              auto hx = peek(3, 6);
              auto input_bias = peek(4, 6);
              auto hidden_bias = peek(5, 6);
              auto the_result = at::_thnn_differentiable_gru_cell_backward(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
                if(OutputSize() > 3) {assignTo(Output(3),std::get<3>(the_result));}
                if(OutputSize() > 4) {assignTo(Output(4),std::get<4>(the_result));}
              return true;
          };
      } break;
      case 700: { // lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 1, InputSize());
              auto params = peekSlice(1, InputSize() - 1, InputSize());
              auto the_result = at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 701: { // lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peekSlice(2, InputSize() - 2, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 702: { // gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 703: { // gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 704: { // rnn_tanh
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 705: { // rnn_tanh
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 706: { // rnn_relu
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 707: { // rnn_relu
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 708: { // lstm_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 5, InputSize());
              auto w_ih = peek(1, 5);
              auto w_hh = peek(2, 5);
              auto b_ih = peek(3, 5);
              auto b_hh = peek(4, 5);
              auto the_result = at::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 709: { // gru_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 6);
              auto hx = peek(1, 6);
              auto w_ih = peek(2, 6);
              auto w_hh = peek(3, 6);
              auto b_ih = peek(4, 6);
              auto b_hh = peek(5, 6);
              auto the_result = at::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 710: { // gru_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto hx = peek(1, 5);
              auto w_ih = peek(2, 5);
              auto w_hh = peek(3, 5);
              auto b_ih = peek(4, 5);
              auto the_result = at::gru_cell(input, hx, w_ih, w_hh, b_ih);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 711: { // gru_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 4);
              auto hx = peek(1, 4);
              auto w_ih = peek(2, 4);
              auto w_hh = peek(3, 4);
              auto the_result = at::gru_cell(input, hx, w_ih, w_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 712: { // rnn_tanh_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 6);
              auto hx = peek(1, 6);
              auto w_ih = peek(2, 6);
              auto w_hh = peek(3, 6);
              auto b_ih = peek(4, 6);
              auto b_hh = peek(5, 6);
              auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 713: { // rnn_tanh_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto hx = peek(1, 5);
              auto w_ih = peek(2, 5);
              auto w_hh = peek(3, 5);
              auto b_ih = peek(4, 5);
              auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 714: { // rnn_tanh_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 4);
              auto hx = peek(1, 4);
              auto w_ih = peek(2, 4);
              auto w_hh = peek(3, 4);
              auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 715: { // rnn_relu_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 6);
              auto hx = peek(1, 6);
              auto w_ih = peek(2, 6);
              auto w_hh = peek(3, 6);
              auto b_ih = peek(4, 6);
              auto b_hh = peek(5, 6);
              auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 716: { // rnn_relu_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 5);
              auto hx = peek(1, 5);
              auto w_ih = peek(2, 5);
              auto w_hh = peek(3, 5);
              auto b_ih = peek(4, 5);
              auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 717: { // rnn_relu_cell
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 4);
              auto hx = peek(1, 4);
              auto w_ih = peek(2, 4);
              auto w_hh = peek(3, 4);
              auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 718: { // quantized_lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 1, InputSize());
              auto params = peekSlice(1, InputSize() - 1, InputSize());
              auto the_result = at::quantized_lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 719: { // quantized_lstm
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peekSlice(2, InputSize() - 2, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::quantized_lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 720: { // quantized_gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peek(1, InputSize());
              auto params = peekSlice(2, InputSize() - 2, InputSize());
              auto the_result = at::quantized_gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 721: { // quantized_gru
          bool has_biases = readAttribute<int64_t>("has_biases");
          int64_t num_layers = readAttribute<int64_t>("num_layers");
          double dropout = readAttribute<float>("dropout");
          bool train = readAttribute<int64_t>("train");
          bool bidirectional = readAttribute<int64_t>("bidirectional");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, InputSize());
              auto batch_sizes = peek(1, InputSize());
              auto hx = peek(2, InputSize());
              auto params = peekSlice(3, InputSize() - 3, InputSize());
              auto the_result = at::quantized_gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 722: { // quantized_lstm_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, InputSize());
              auto hx = peekSlice(1, InputSize() - 9, InputSize());
              auto w_ih = peek(1, 9);
              auto w_hh = peek(2, 9);
              auto b_ih = peek(3, 9);
              auto b_hh = peek(4, 9);
              auto packed_ih = peek(5, 9);
              auto packed_hh = peek(6, 9);
              auto col_offsets_ih = peek(7, 9);
              auto col_offsets_hh = peek(8, 9);
              auto the_result = at::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 723: { // quantized_gru_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 10);
              auto hx = peek(1, 10);
              auto w_ih = peek(2, 10);
              auto w_hh = peek(3, 10);
              auto b_ih = peek(4, 10);
              auto b_hh = peek(5, 10);
              auto packed_ih = peek(6, 10);
              auto packed_hh = peek(7, 10);
              auto col_offsets_ih = peek(8, 10);
              auto col_offsets_hh = peek(9, 10);
              auto the_result = at::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 724: { // quantized_rnn_relu_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 10);
              auto hx = peek(1, 10);
              auto w_ih = peek(2, 10);
              auto w_hh = peek(3, 10);
              auto b_ih = peek(4, 10);
              auto b_hh = peek(5, 10);
              auto packed_ih = peek(6, 10);
              auto packed_hh = peek(7, 10);
              auto col_offsets_ih = peek(8, 10);
              auto col_offsets_hh = peek(9, 10);
              auto the_result = at::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 725: { // quantized_rnn_tanh_cell
          at::Scalar scale_ih = readScalarAttribute("scale_ih");
          at::Scalar scale_hh = readScalarAttribute("scale_hh");
          at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
          at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 10);
              auto hx = peek(1, 10);
              auto w_ih = peek(2, 10);
              auto w_hh = peek(3, 10);
              auto b_ih = peek(4, 10);
              auto b_hh = peek(5, 10);
              auto packed_ih = peek(6, 10);
              auto packed_hh = peek(7, 10);
              auto col_offsets_ih = peek(8, 10);
              auto col_offsets_hh = peek(9, 10);
              auto the_result = at::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 726: { // _pack_padded_sequence
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto input = peek(0, 2);
              auto lengths = peek(1, 2);
              auto the_result = at::_pack_padded_sequence(input, lengths, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 727: { // _pack_padded_sequence_backward
          auto input_size = readIntArrayRef("input_size");
          bool batch_first = readAttribute<int64_t>("batch_first");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad = peek(0, 2);
              auto batch_sizes = peek(1, 2);
              auto the_result = at::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 728: { // _pad_packed_sequence
          bool batch_first = readAttribute<int64_t>("batch_first");
          at::Scalar padding_value = readScalarAttribute("padding_value");
          int64_t total_length = readAttribute<int64_t>("total_length");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto data = peek(0, 2);
              auto batch_sizes = peek(1, 2);
              auto the_result = at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 729: { // is_set_to
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto tensor = peek(1, 2);
              auto the_result = self.is_set_to(tensor);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 730: { // masked_fill
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = at::masked_fill(self, mask, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 731: { // masked_fill
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mask = peek(1, 3);
              auto value = peek(2, 3);
              auto the_result = at::masked_fill(self, mask, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 732: { // masked_scatter
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto mask = peek(1, 3);
              auto source = peek(2, 3);
              auto the_result = at::masked_scatter(self, mask, source);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 733: { // view
          auto size = readIntArrayRef("size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.view(size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 734: { // index_add
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto source = peek(2, 3);
              auto the_result = at::index_add(self, dim, index, source);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 735: { // index_fill
          int64_t dim = readAttribute<int64_t>("dim");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::index_fill(self, dim, index, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 736: { // index_fill
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto value = peek(2, 3);
              auto the_result = at::index_fill(self, dim, index, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 737: { // scatter
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto src = peek(2, 3);
              auto the_result = at::scatter(self, dim, index, src);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 738: { // scatter
          int64_t dim = readAttribute<int64_t>("dim");
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::scatter(self, dim, index, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 739: { // scatter_add
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto src = peek(2, 3);
              auto the_result = at::scatter_add(self, dim, index, src);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 740: { // bitwise_and
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bitwise_and(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 741: { // bitwise_and
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::bitwise_and(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 742: { // __and__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::__and__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 743: { // __and__
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__and__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 744: { // bitwise_or
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bitwise_or(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 745: { // bitwise_or
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::bitwise_or(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 746: { // __or__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::__or__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 747: { // __or__
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__or__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 748: { // bitwise_xor
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::bitwise_xor(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 749: { // bitwise_xor
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::bitwise_xor(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 750: { // __xor__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::__xor__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 751: { // __xor__
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__xor__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 752: { // __lshift__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::__lshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 753: { // __lshift__
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__lshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 754: { // __rshift__
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::__rshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 755: { // __rshift__
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::__rshift__(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 756: { // addbmm
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::addbmm(self, batch1, batch2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 757: { // addbmm
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::addbmm(self, batch1, batch2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 758: { // addbmm
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto batch1 = peek(1, 3);
              auto batch2 = peek(2, 3);
              auto the_result = at::addbmm(self, batch1, batch2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 759: { // diag
          int64_t diagonal = readAttribute<int64_t>("diagonal");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diag(self, diagonal);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 760: { // diag
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::diag(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 761: { // cross
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::cross(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 762: { // triu
          int64_t diagonal = readAttribute<int64_t>("diagonal");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::triu(self, diagonal);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 763: { // triu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::triu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 764: { // tril
          int64_t diagonal = readAttribute<int64_t>("diagonal");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::tril(self, diagonal);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 765: { // tril
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::tril(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 766: { // trace
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::trace(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 767: { // ne
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::ne(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 768: { // ne
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::ne(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 769: { // eq
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::eq(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 770: { // eq
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::eq(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 771: { // ge
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::ge(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 772: { // ge
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::ge(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 773: { // le
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::le(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 774: { // le
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::le(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 775: { // gt
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::gt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 776: { // gt
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::gt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 777: { // lt
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::lt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 778: { // lt
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::lt(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 779: { // take
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::take(self, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 780: { // index_select
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::index_select(self, dim, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 781: { // masked_select
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto mask = peek(1, 2);
              auto the_result = at::masked_select(self, mask);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 782: { // nonzero
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::nonzero(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 783: { // nonzero_numpy
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::nonzero_numpy(self);
                if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
              return true;
          };
      } break;
      case 784: { // gather
          int64_t dim = readAttribute<int64_t>("dim");
          bool sparse_grad = readAttribute<int64_t>("sparse_grad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::gather(self, dim, index, sparse_grad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 785: { // gather
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto index = peek(1, 2);
              auto the_result = at::gather(self, dim, index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 786: { // _gather_sparse_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto index = peek(1, 3);
              auto grad = peek(2, 3);
              auto the_result = at::_gather_sparse_backward(self, dim, index, grad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 787: { // addcmul
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcmul(self, tensor1, tensor2, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 788: { // addcmul
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcmul(self, tensor1, tensor2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 789: { // addcdiv
          at::Scalar value = readScalarAttribute("value");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcdiv(self, tensor1, tensor2, value);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 790: { // addcdiv
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto tensor1 = peek(1, 3);
              auto tensor2 = peek(2, 3);
              auto the_result = at::addcdiv(self, tensor1, tensor2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 791: { // lstsq
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::lstsq(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 792: { // triangular_solve
          bool upper = readAttribute<int64_t>("upper");
          bool transpose = readAttribute<int64_t>("transpose");
          bool unitriangular = readAttribute<int64_t>("unitriangular");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A, upper, transpose, unitriangular);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 793: { // triangular_solve
          bool upper = readAttribute<int64_t>("upper");
          bool transpose = readAttribute<int64_t>("transpose");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A, upper, transpose);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 794: { // triangular_solve
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A, upper);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 795: { // triangular_solve
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::triangular_solve(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 796: { // _triangular_solve_helper
          bool upper = readAttribute<int64_t>("upper");
          bool transpose = readAttribute<int64_t>("transpose");
          bool unitriangular = readAttribute<int64_t>("unitriangular");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::_triangular_solve_helper(self, A, upper, transpose, unitriangular);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 797: { // symeig
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::symeig(self, eigenvectors, upper);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 798: { // symeig
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::symeig(self, eigenvectors);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 799: { // symeig
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::symeig(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 800: { // _symeig_helper
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_symeig_helper(self, eigenvectors, upper);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 801: { // eig
          bool eigenvectors = readAttribute<int64_t>("eigenvectors");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::eig(self, eigenvectors);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 802: { // eig
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::eig(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 803: { // svd
          bool some = readAttribute<int64_t>("some");
          bool compute_uv = readAttribute<int64_t>("compute_uv");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::svd(self, some, compute_uv);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 804: { // svd
          bool some = readAttribute<int64_t>("some");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::svd(self, some);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 805: { // svd
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::svd(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 806: { // _svd_helper
          bool some = readAttribute<int64_t>("some");
          bool compute_uv = readAttribute<int64_t>("compute_uv");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_svd_helper(self, some, compute_uv);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 807: { // cholesky
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cholesky(self, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 808: { // cholesky
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cholesky(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 809: { // _cholesky_helper
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cholesky_helper(self, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 810: { // cholesky_solve
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto input2 = peek(1, 2);
              auto the_result = at::cholesky_solve(self, input2, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 811: { // cholesky_solve
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto input2 = peek(1, 2);
              auto the_result = at::cholesky_solve(self, input2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 812: { // _cholesky_solve_helper
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::_cholesky_solve_helper(self, A, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 813: { // solve
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::solve(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 814: { // _solve_helper
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto A = peek(1, 2);
              auto the_result = at::_solve_helper(self, A);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 815: { // cholesky_inverse
          bool upper = readAttribute<int64_t>("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cholesky_inverse(self, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 816: { // cholesky_inverse
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::cholesky_inverse(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 817: { // qr
          bool some = readAttribute<int64_t>("some");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::qr(self, some);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 818: { // qr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::qr(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 819: { // _qr_helper
          bool some = readAttribute<int64_t>("some");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_qr_helper(self, some);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 820: { // geqrf
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::geqrf(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 821: { // orgqr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto input2 = peek(1, 2);
              auto the_result = at::orgqr(self, input2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 822: { // ormqr
          bool left = readAttribute<int64_t>("left");
          bool transpose = readAttribute<int64_t>("transpose");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto input2 = peek(1, 3);
              auto input3 = peek(2, 3);
              auto the_result = at::ormqr(self, input2, input3, left, transpose);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 823: { // ormqr
          bool left = readAttribute<int64_t>("left");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto input2 = peek(1, 3);
              auto input3 = peek(2, 3);
              auto the_result = at::ormqr(self, input2, input3, left);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 824: { // ormqr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto input2 = peek(1, 3);
              auto input3 = peek(2, 3);
              auto the_result = at::ormqr(self, input2, input3);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 825: { // _lu_with_info
          bool pivot = readAttribute<int64_t>("pivot");
          bool check_errors = readAttribute<int64_t>("check_errors");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_lu_with_info(self, pivot, check_errors);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 826: { // _lu_with_info
          bool pivot = readAttribute<int64_t>("pivot");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_lu_with_info(self, pivot);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 827: { // _lu_with_info
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_lu_with_info(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 828: { // lu_solve
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto LU_data = peek(1, 3);
              auto LU_pivots = peek(2, 3);
              auto the_result = at::lu_solve(self, LU_data, LU_pivots);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 829: { // _lu_solve_helper
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto LU_data = peek(1, 3);
              auto LU_pivots = peek(2, 3);
              auto the_result = at::_lu_solve_helper(self, LU_data, LU_pivots);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 830: { // multinomial
          int64_t num_samples = readAttribute<int64_t>("num_samples");
          bool replacement = readAttribute<int64_t>("replacement");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::multinomial(self, num_samples, replacement);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 831: { // multinomial
          int64_t num_samples = readAttribute<int64_t>("num_samples");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::multinomial(self, num_samples);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 832: { // _multinomial_alias_setup
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto probs = peek(0, 1);
              auto the_result = at::_multinomial_alias_setup(probs);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 833: { // _multinomial_alias_draw
          int64_t num_samples = readAttribute<int64_t>("num_samples");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto J = peek(0, 2);
              auto q = peek(1, 2);
              auto the_result = at::_multinomial_alias_draw(J, q, num_samples);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 834: { // lgamma
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::lgamma(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 835: { // digamma
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::digamma(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 836: { // polygamma
          int64_t n = readAttribute<int64_t>("n");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::polygamma(n, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 837: { // erfinv
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::erfinv(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 838: { // sign
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sign(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 839: { // dist
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::dist(self, other, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 840: { // dist
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::dist(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 841: { // atan2
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::atan2(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 842: { // lerp
          at::Scalar weight = readScalarAttribute("weight");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto end = peek(1, 2);
              auto the_result = at::lerp(self, end, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 843: { // lerp
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto end = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::lerp(self, end, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 844: { // histc
          int64_t bins = readAttribute<int64_t>("bins");
          at::Scalar min = readScalarAttribute("min");
          at::Scalar max = readScalarAttribute("max");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::histc(self, bins, min, max);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 845: { // histc
          int64_t bins = readAttribute<int64_t>("bins");
          at::Scalar min = readScalarAttribute("min");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::histc(self, bins, min);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 846: { // histc
          int64_t bins = readAttribute<int64_t>("bins");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::histc(self, bins);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 847: { // histc
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::histc(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 848: { // fmod
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::fmod(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 849: { // fmod
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::fmod(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 850: { // remainder
          at::Scalar other = readScalarAttribute("other");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::remainder(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 851: { // remainder
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::remainder(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 852: { // min
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::min(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 853: { // min
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::min(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 854: { // max
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::max(self, other);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 855: { // max
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 856: { // median
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::median(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 857: { // sort
          int64_t dim = readAttribute<int64_t>("dim");
          bool descending = readAttribute<int64_t>("descending");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sort(self, dim, descending);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 858: { // sort
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sort(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 859: { // sort
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::sort(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 860: { // argsort
          int64_t dim = readAttribute<int64_t>("dim");
          bool descending = readAttribute<int64_t>("descending");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::argsort(self, dim, descending);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 861: { // argsort
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::argsort(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 862: { // argsort
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::argsort(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 863: { // topk
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          bool largest = readAttribute<int64_t>("largest");
          bool sorted = readAttribute<int64_t>("sorted");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k, dim, largest, sorted);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 864: { // topk
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          bool largest = readAttribute<int64_t>("largest");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k, dim, largest);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 865: { // topk
          int64_t k = readAttribute<int64_t>("k");
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 866: { // topk
          int64_t k = readAttribute<int64_t>("k");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::topk(self, k);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 867: { // all
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::all(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 868: { // any
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::any(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 869: { // renorm
          at::Scalar p = readScalarAttribute("p");
          int64_t dim = readAttribute<int64_t>("dim");
          at::Scalar maxnorm = readScalarAttribute("maxnorm");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::renorm(self, p, dim, maxnorm);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 870: { // unfold
          int64_t dimension = readAttribute<int64_t>("dimension");
          int64_t size = readAttribute<int64_t>("size");
          int64_t step = readAttribute<int64_t>("step");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = self.unfold(dimension, size, step);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 871: { // equal
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto other = peek(1, 2);
              auto the_result = at::equal(self, other);
                if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
              return true;
          };
      } break;
      case 872: { // pow
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto exponent = peek(1, 2);
              auto the_result = at::pow(self, exponent);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 873: { // pow
          at::Scalar self = readScalarAttribute("self");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto exponent = peek(0, 1);
              auto the_result = at::pow(self, exponent);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 874: { // alias
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::alias(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 875: { // _addr
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::_addr(self, vec1, vec2, beta, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 876: { // _addr
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::_addr(self, vec1, vec2, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 877: { // _addr
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto vec1 = peek(1, 3);
              auto vec2 = peek(2, 3);
              auto the_result = at::_addr(self, vec1, vec2);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 878: { // _cumsum
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cumsum(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 879: { // _cumprod
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_cumprod(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 880: { // _var
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_var(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 881: { // _var
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_var(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 882: { // _std
          bool unbiased = readAttribute<int64_t>("unbiased");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_std(self, unbiased);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 883: { // _std
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_std(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 884: { // _cat
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::_cat(tensors, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 885: { // _cat
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto tensors = peekSlice(0, InputSize() - 0, InputSize());
              auto the_result = at::_cat(tensors);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 886: { // _mode
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_mode(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 887: { // _mode
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_mode(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 888: { // _mode
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_mode(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 889: { // _max
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_max(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 890: { // _max
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_max(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 891: { // _min
          int64_t dim = readAttribute<int64_t>("dim");
          bool keepdim = readAttribute<int64_t>("keepdim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_min(self, dim, keepdim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 892: { // _min
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_min(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 893: { // mse_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::mse_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 894: { // mse_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::mse_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 895: { // mse_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::mse_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 896: { // l1_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::l1_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 897: { // l1_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::l1_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 898: { // l1_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::l1_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 899: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::multi_margin_loss(self, target, p, margin, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 900: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::multi_margin_loss(self, target, p, margin, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 901: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multi_margin_loss(self, target, p, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 902: { // multi_margin_loss
          at::Scalar p = readScalarAttribute("p");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multi_margin_loss(self, target, p);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 903: { // multi_margin_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multi_margin_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 904: { // multi_margin_loss_backward
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 905: { // multi_margin_loss_backward
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto weight = peek(3, 4);
              auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 906: { // multi_margin_loss_backward
          at::Scalar p = readScalarAttribute("p");
          at::Scalar margin = readScalarAttribute("margin");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 907: { // multilabel_margin_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multilabel_margin_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 908: { // multilabel_margin_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multilabel_margin_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 909: { // multilabel_margin_loss_forward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::multilabel_margin_loss_forward(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 910: { // multilabel_margin_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 4);
              auto self = peek(1, 4);
              auto target = peek(2, 4);
              auto is_target = peek(3, 4);
              auto the_result = at::multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 911: { // nll_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 912: { // nll_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss(self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 913: { // nll_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 914: { // nll_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::nll_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 915: { // nll_loss_forward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss_forward(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 916: { // nll_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto total_weight = peek(4, 5);
              auto the_result = at::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 917: { // nll_loss2d
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 918: { // nll_loss2d
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d(self, target, weight, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 919: { // nll_loss2d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d(self, target, weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 920: { // nll_loss2d
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::nll_loss2d(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 921: { // nll_loss2d_forward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto target = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 922: { // nll_loss2d_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          int64_t ignore_index = readAttribute<int64_t>("ignore_index");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto target = peek(2, 5);
              auto weight = peek(3, 5);
              auto total_weight = peek(4, 5);
              auto the_result = at::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 923: { // smooth_l1_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::smooth_l1_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 924: { // smooth_l1_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::smooth_l1_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 925: { // smooth_l1_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::smooth_l1_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 926: { // soft_margin_loss
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::soft_margin_loss(self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 927: { // soft_margin_loss
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto target = peek(1, 2);
              auto the_result = at::soft_margin_loss(self, target);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 928: { // soft_margin_loss_backward
          int64_t reduction = readAttribute<int64_t>("reduction");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto target = peek(2, 3);
              auto the_result = at::soft_margin_loss_backward(grad_output, self, target, reduction);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 929: { // elu
          at::Scalar alpha = readScalarAttribute("alpha");
          at::Scalar scale = readScalarAttribute("scale");
          at::Scalar input_scale = readScalarAttribute("input_scale");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::elu(self, alpha, scale, input_scale);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 930: { // elu
          at::Scalar alpha = readScalarAttribute("alpha");
          at::Scalar scale = readScalarAttribute("scale");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::elu(self, alpha, scale);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 931: { // elu
          at::Scalar alpha = readScalarAttribute("alpha");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::elu(self, alpha);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 932: { // elu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::elu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 933: { // elu_backward
          at::Scalar alpha = readScalarAttribute("alpha");
          at::Scalar scale = readScalarAttribute("scale");
          at::Scalar input_scale = readScalarAttribute("input_scale");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::elu_backward(grad_output, alpha, scale, input_scale, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 934: { // glu
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::glu(self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 935: { // glu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::glu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 936: { // glu_backward
          int64_t dim = readAttribute<int64_t>("dim");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::glu_backward(grad_output, self, dim);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 937: { // hardtanh
          at::Scalar min_val = readScalarAttribute("min_val");
          at::Scalar max_val = readScalarAttribute("max_val");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::hardtanh(self, min_val, max_val);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 938: { // hardtanh
          at::Scalar min_val = readScalarAttribute("min_val");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::hardtanh(self, min_val);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 939: { // hardtanh
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::hardtanh(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 940: { // hardtanh_backward
          at::Scalar min_val = readScalarAttribute("min_val");
          at::Scalar max_val = readScalarAttribute("max_val");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::hardtanh_backward(grad_output, self, min_val, max_val);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 941: { // leaky_relu
          at::Scalar negative_slope = readScalarAttribute("negative_slope");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::leaky_relu(self, negative_slope);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 942: { // leaky_relu
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::leaky_relu(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 943: { // leaky_relu_backward
          at::Scalar negative_slope = readScalarAttribute("negative_slope");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::leaky_relu_backward(grad_output, self, negative_slope);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 944: { // log_sigmoid
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log_sigmoid(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 945: { // log_sigmoid_forward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::log_sigmoid_forward(self);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 946: { // log_sigmoid_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto buffer = peek(2, 3);
              auto the_result = at::log_sigmoid_backward(grad_output, self, buffer);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 947: { // rrelu_with_noise
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          bool training = readAttribute<int64_t>("training");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise, lower, upper, training);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 948: { // rrelu_with_noise
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise, lower, upper);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 949: { // rrelu_with_noise
          at::Scalar lower = readScalarAttribute("lower");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise, lower);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 950: { // rrelu_with_noise
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto noise = peek(1, 2);
              auto the_result = at::rrelu_with_noise(self, noise);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 951: { // rrelu_with_noise_backward
          at::Scalar lower = readScalarAttribute("lower");
          at::Scalar upper = readScalarAttribute("upper");
          bool training = readAttribute<int64_t>("training");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto noise = peek(2, 3);
              auto the_result = at::rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 952: { // softplus
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar threshold = readScalarAttribute("threshold");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::softplus(self, beta, threshold);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 953: { // softplus
          at::Scalar beta = readScalarAttribute("beta");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::softplus(self, beta);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 954: { // softplus
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::softplus(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 955: { // softplus_backward
          at::Scalar beta = readScalarAttribute("beta");
          at::Scalar threshold = readScalarAttribute("threshold");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto output = peek(2, 3);
              auto the_result = at::softplus_backward(grad_output, self, beta, threshold, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 956: { // softshrink
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::softshrink(self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 957: { // softshrink
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::softshrink(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 958: { // softshrink_backward
          at::Scalar lambd = readScalarAttribute("lambd");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::softshrink_backward(grad_output, self, lambd);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 959: { // adaptive_avg_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::adaptive_avg_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 960: { // mkldnn_adaptive_avg_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::mkldnn_adaptive_avg_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 961: { // _adaptive_avg_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::_adaptive_avg_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 962: { // _adaptive_avg_pool2d_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::_adaptive_avg_pool2d_backward(grad_output, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 963: { // adaptive_avg_pool3d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::adaptive_avg_pool3d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 964: { // adaptive_avg_pool3d_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::adaptive_avg_pool3d_backward(grad_output, self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 965: { // adaptive_max_pool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::adaptive_max_pool2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 966: { // adaptive_max_pool2d_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::adaptive_max_pool2d_backward(grad_output, self, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 967: { // adaptive_max_pool3d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::adaptive_max_pool3d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 968: { // adaptive_max_pool3d_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::adaptive_max_pool3d_backward(grad_output, self, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 969: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          bool count_include_pad = readAttribute<int64_t>("count_include_pad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 970: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 971: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 972: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 973: { // avg_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool2d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 974: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          bool count_include_pad = readAttribute<int64_t>("count_include_pad");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 975: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 976: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 977: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 978: { // avg_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::avg_pool3d(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 979: { // fractional_max_pool2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto random_samples = peek(1, 2);
              auto the_result = at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 980: { // fractional_max_pool2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 981: { // fractional_max_pool3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto random_samples = peek(1, 2);
              auto the_result = at::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 982: { // fractional_max_pool3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::fractional_max_pool3d_backward(grad_output, self, kernel_size, output_size, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 983: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 984: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 985: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 986: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 987: { // max_pool2d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool2d_with_indices(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 988: { // max_pool2d_with_indices_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 989: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 990: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 991: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 992: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 993: { // max_pool3d_with_indices
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::max_pool3d_with_indices(self, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 994: { // max_pool3d_with_indices_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          bool ceil_mode = readAttribute<int64_t>("ceil_mode");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 995: { // max_unpool2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::max_unpool2d(self, indices, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 996: { // max_unpool2d_backward
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_unpool2d_backward(grad_output, self, indices, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 997: { // max_unpool3d
          auto output_size = readIntArrayRef("output_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto indices = peek(1, 2);
              auto the_result = at::max_unpool3d(self, indices, output_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 998: { // max_unpool3d_backward
          auto output_size = readIntArrayRef("output_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto indices = peek(2, 3);
              auto the_result = at::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 999: { // reflection_pad1d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::reflection_pad1d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1000: { // reflection_pad1d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::reflection_pad1d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1001: { // reflection_pad2d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::reflection_pad2d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1002: { // reflection_pad2d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::reflection_pad2d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1003: { // replication_pad1d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::replication_pad1d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1004: { // replication_pad1d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::replication_pad1d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1005: { // replication_pad2d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::replication_pad2d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1006: { // replication_pad2d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::replication_pad2d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1007: { // replication_pad3d
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::replication_pad3d(self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1008: { // replication_pad3d_backward
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto self = peek(1, 2);
              auto the_result = at::replication_pad3d_backward(grad_output, self, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1009: { // upsample_linear1d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_linear1d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1010: { // upsample_linear1d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1011: { // upsample_bilinear2d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_bilinear2d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1012: { // upsample_bilinear2d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1013: { // upsample_bicubic2d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_bicubic2d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1014: { // upsample_bicubic2d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1015: { // upsample_trilinear3d
          auto output_size = readIntArrayRef("output_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_trilinear3d(self, output_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1016: { // upsample_trilinear3d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          bool align_corners = readAttribute<int64_t>("align_corners");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1017: { // upsample_nearest1d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_nearest1d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1018: { // upsample_nearest1d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_nearest1d_backward(grad_output, output_size, input_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1019: { // upsample_nearest2d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_nearest2d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1020: { // upsample_nearest2d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_nearest2d_backward(grad_output, output_size, input_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1021: { // upsample_nearest3d
          auto output_size = readIntArrayRef("output_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::upsample_nearest3d(self, output_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1022: { // upsample_nearest3d_backward
          auto output_size = readIntArrayRef("output_size");
          auto input_size = readIntArrayRef("input_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::upsample_nearest3d_backward(grad_output, output_size, input_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1023: { // sigmoid_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::sigmoid_backward(grad_output, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1024: { // tanh_backward
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 2);
              auto output = peek(1, 2);
              auto the_result = at::tanh_backward(grad_output, output);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1025: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1026: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1027: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1028: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1029: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1030: { // slow_conv_transpose2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1031: { // slow_conv_transpose2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto columns = peek(3, 5);
              auto ones = peek(4, 5);
              auto the_result = at::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1032: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1033: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1034: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1035: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1036: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1037: { // slow_conv_transpose3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1038: { // slow_conv_transpose3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_padding = readIntArrayRef("output_padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto finput = peek(3, 5);
              auto fgrad_input = peek(4, 5);
              auto the_result = at::slow_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1039: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1040: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1041: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1042: { // thnn_conv2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::thnn_conv2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1043: { // thnn_conv2d_forward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1044: { // thnn_conv2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto finput = peek(3, 5);
              auto fgrad_input = peek(4, 5);
              auto the_result = at::thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1045: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1046: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1047: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1048: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1049: { // thnn_conv_depthwise2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::thnn_conv_depthwise2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1050: { // thnn_conv_depthwise2d_forward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1051: { // thnn_conv_depthwise2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<2>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
              return true;
          };
      } break;
      case 1052: { // slow_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1053: { // slow_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv3d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1054: { // slow_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv3d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1055: { // slow_conv3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv3d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1056: { // slow_conv3d_forward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1057: { // slow_conv3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 5);
              auto self = peek(1, 5);
              auto weight = peek(2, 5);
              auto finput = peek(3, 5);
              auto fgrad_input = peek(4, 5);
              auto the_result = at::slow_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1058: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1059: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1060: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1061: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1062: { // slow_conv_dilated2d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1063: { // slow_conv_dilated2d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1064: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1065: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1066: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1067: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 3);
              auto weight = peek(1, 3);
              auto bias = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1068: { // slow_conv_dilated3d
          auto kernel_size = readIntArrayRef("kernel_size");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 2);
              auto weight = peek(1, 2);
              auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1069: { // slow_conv_dilated3d_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto stride = readIntArrayRef("stride");
          auto padding = readIntArrayRef("padding");
          auto dilation = readIntArrayRef("dilation");
          auto output_mask = readBoolMask<3>("output_mask");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 3);
              auto self = peek(1, 3);
              auto weight = peek(2, 3);
              auto the_result = at::slow_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
                if(OutputSize() > 0) {assignTo(Output(0),std::get<0>(the_result));}
                if(OutputSize() > 1) {assignTo(Output(1),std::get<1>(the_result));}
                if(OutputSize() > 2) {assignTo(Output(2),std::get<2>(the_result));}
              return true;
          };
      } break;
      case 1070: { // col2im
          auto output_size = readIntArrayRef("output_size");
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::col2im(self, output_size, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1071: { // col2im_backward
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::col2im_backward(grad_output, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1072: { // im2col
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::im2col(self, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1073: { // im2col_backward
          auto input_size = readIntArrayRef("input_size");
          auto kernel_size = readIntArrayRef("kernel_size");
          auto dilation = readIntArrayRef("dilation");
          auto padding = readIntArrayRef("padding");
          auto stride = readIntArrayRef("stride");
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto grad_output = peek(0, 1);
              auto the_result = at::im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1074: { // isfinite
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::isfinite(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      case 1075: { // isinf
      
          run_op = [=] {
              at::AutoNonVariableTypeMode guard;
              auto self = peek(0, 1);
              auto the_result = at::isinf(self);
                if(OutputSize() > 0) {assignTo(Output(0),the_result);}
              return true;
          };
      } break;
      default:
        CAFFE_THROW("Unexpected key value for aten operator");
    }
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return run_op();
  }
private:
  // actual operator implementation is initialized in ctor.
  std::function<bool()> run_op;
  at::Backend backend() const;

  TypeMeta typeMetaFor(const at::Tensor & t) {
    return typeMetaFor(t.scalar_type());
  }
  TypeMeta typeMetaFor(at::ScalarType st) {
    #define DEFINE_CASE(ctype,aten_name) \
      case at::k##aten_name: \
        return TypeMeta::Make<ctype>();
    switch(st) {
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
    default:
      CAFFE_THROW("Unknown ATen Type");
    }
    #undef DEFINE_CASE
  }

  at::TensorOptions optionsFor(const Tensor& ten) {
    at::Device device = ten.GetDevice();
#ifdef __HIP_PLATFORM_HCC__
    if (backend() == at::Backend::HIP) {
      device = at::Device(kCUDA, device.index());
    }
#endif
    return at::TensorOptions(device).dtype(ten.dtype());
  }

  at::Tensor tensorWrapping(const Tensor& ten_) {
    auto& ten = const_cast<Tensor&>(ten_);
    return at::from_blob(
        ten.raw_mutable_data(),
        ten.sizes(),
        optionsFor(ten));
  }

  at::Tensor peek(size_t i, size_t N) {
    auto real_idx = InputSize() - N + i;
    return tensorWrapping(Input(real_idx));
  }

  std::vector<at::Tensor> peekSlice(size_t i, size_t len, size_t N) {
    std::vector<at::Tensor> results;
    for (size_t ii = i; ii < i + len; ++ii) {
      results.push_back(peek(ii, N));
    }
    return results;
  }

  void assignTo(Tensor* dst, const at::Tensor& src_) {
    at::Tensor src = src_.contiguous();
    auto at_sizes = src.sizes();
    caffe2::TypeMeta type_meta = typeMetaFor(src);
    at::Device device = src.device();
#ifdef __HIP_PLATFORM_HCC__
    if (device.type() == at::DeviceType::CUDA) {
      device = at::Device(at::DeviceType::HIP, device.index());
    }
#endif
    at::TensorImpl* src_impl = src.unsafeReleaseTensorImpl();
    std::vector<int64_t> dims(at_sizes.begin(), at_sizes.end());
    dst->Resize(dims);
    dst->ShareExternalPointer(
        at::DataPtr(
            src_impl->data(),
            static_cast<void*>(src_impl),
            [](void* t_ptr) -> void {
              at::TensorImpl* local_impl = static_cast<at::TensorImpl*>(t_ptr);
              c10::raw::intrusive_ptr::decref(local_impl);
            },
            device),
        type_meta,
        0);
  }
  void assignListStartingAt(
      size_t offset,
      const std::vector<at::Tensor>& tensors) {
    for (size_t i = 0; i < tensors.size(); i++) {
      assignTo(Output(offset + i), tensors[i]);
    }
  }

  template<typename T,
          typename std::enable_if<std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toLong();
  }

  template<typename T,
          typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toDouble();
  }

  void assignTo(Tensor* dst, at::ScalarType scalar_type, at::Scalar scalar) {
    switch(scalar_type) {
      #define DEFINE_CASE(ctype,aten_name) \
        case at::k##aten_name: { \
          auto value = extract<ctype>(scalar); \
          assignToValue<ctype>(dst, at::convert<ctype,decltype(value)>(value)); \
        } break;
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
#undef DEFINE_CASE
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
  }
  template <typename T>
  void assignToValue(Tensor* dst, T v) {
    dst->Resize(std::vector<int64_t>());
    math::Set(1, v, dst->template mutable_data<T>(), &context_);
  }
  int findImplementation(const OperatorDef& operator_def) {
    CAFFE_ENFORCE(HasArgument("operator"));
    std::string op = OperatorBase::GetSingleArgument<std::string>("operator", "");
    // construct descriptor string ([DESCRIPTORS]) given the attributes
    // and inputs of this operator_def, and look up the implementation key
    // for this variant
    std::stringstream descriptor;
    descriptor << op;
    std::vector<std::string> attrs;
    for(size_t i = 0; i < operator_def.arg_size(); i++) {
      auto & attr = operator_def.arg(i);
      if(attr.name() == "operator" || attr.name() == "type" )
        continue;
      attrs.push_back(attr.name());
    }
    std::sort(attrs.begin(), attrs.end());
    for(auto & a : attrs)
      descriptor << "-" << a;

    std::string descriptor_sized =
        descriptor.str() + "-" + c10::to_string(InputSize());
    std::string descriptor_var_args = descriptor.str() + "-*";
    if (op_to_key.count(descriptor_sized) > 0) {
      return op_to_key[descriptor_sized];
    }
    if (op_to_key.count(descriptor_var_args) > 0) {
      return op_to_key[descriptor_var_args];
    }
    std::stringstream ss;
    ss << "Attempting to run unknown ATen operator configuration: "
       << descriptor_sized;
    CAFFE_THROW(ss.str());
  }
  at::Scalar readScalarAttribute(const std::string & name) {
    if(OperatorBase::HasSingleArgumentOfType<int64_t>(name)) {
      return OperatorBase::GetSingleArgument<int64_t>(name, 0);
    } else {
      CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<float>(name));
      return OperatorBase::GetSingleArgument<float>(name, 0);
    }
  }
  template<typename T>
  T readAttribute(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<T>(name));
    return OperatorBase::GetSingleArgument<T>(name, 0);
  }
  std::vector<int64_t> readIntArrayRef(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    return OperatorBase::GetRepeatedArgument<int64_t>(name, {});
  }
  template <int N>
  std::array<bool, N> readBoolMask(const std::string& name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    std::vector<int64_t> ints =
        OperatorBase::GetRepeatedArgument<int64_t>(name, {});
    std::array<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = ints.at(i);
    }
    return result;
  }
};

}
