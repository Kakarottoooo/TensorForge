# TensorForge profiler summary

Classification: **kernel-launch/latency-dominant candidate** (provisional).

torch.profiler nonexclusive self-device-time shares: matrix multiply 25.3%, memory/pointwise 42.2%, short repeated kernels 59.9%. Launch candidates overlap operator categories.

Required confirmation: Use Nsight Compute counters and arithmetic-intensity/roofline analysis; profiler timings alone cannot distinguish DRAM bandwidth from dependency or launch stalls.

| Operator | Category | Calls | Self device (us) | Avg self device (us) | Self CPU (us) |
|---|---|---:|---:|---:|---:|
| `tensorforge::decode` | region | 1 | 38529.000 | 38529.000 | 55379.500 |
| `tensorforge::prefill` | region | 1 | 10475.000 | 10475.000 | 14186.100 |
| `aten::empty_strided` | other | 500 | 6821.000 | 13.642 | 1509.100 |
| `aten::bmm` | matrix_multiply | 8 | 5021.000 | 627.625 | 5000.800 |
| `aten::cat` | memory_or_layout | 125 | 2772.000 | 22.176 | 2396.600 |
| `aten::bmm` | matrix_multiply | 8 | 2709.000 | 338.625 | 2696.600 |
| `aten::empty` | other | 245 | 2234.000 | 9.118 | 817.200 |
| `aten::_to_copy` | memory_or_layout | 93 | 2106.000 | 22.645 | 3247.300 |
| `aten::to` | memory_or_layout | 93 | 1917.000 | 20.613 | 1361.300 |
| `aten::mm` | matrix_multiply | 1 | 1917.000 | 1917.000 | 231.000 |
| `aten::copy_` | memory_or_layout | 93 | 1809.000 | 19.452 | 1319.400 |
| `aten::t` | other | 80 | 1651.000 | 20.637 | 1498.100 |
| `aten::t` | other | 80 | 1458.000 | 18.225 | 1188.300 |
| `aten::bmm` | matrix_multiply | 8 | 1444.000 | 180.500 | 1373.200 |
| `aten::copy_` | memory_or_layout | 51 | 1349.000 | 26.451 | 667.000 |
| `aten::t` | other | 80 | 1241.000 | 15.512 | 1212.100 |
| `aten::matmul` | matrix_multiply | 16 | 1236.000 | 77.250 | 935.400 |
| `aten::repeat_interleave` | other | 16 | 1234.000 | 77.125 | 974.100 |
| `aten::_to_copy` | memory_or_layout | 51 | 1228.000 | 24.078 | 1551.300 |
| `aten::silu` | pointwise | 8 | 1228.000 | 153.500 | 128.700 |
| `aten::transpose` | memory_or_layout | 80 | 1223.000 | 15.287 | 1364.800 |
| `aten::ones` | other | 40 | 1208.000 | 30.200 | 1351.400 |
| `aten::copy_` | memory_or_layout | 51 | 1202.000 | 23.569 | 938.100 |
| `aten::_to_copy` | memory_or_layout | 51 | 1193.000 | 23.392 | 1611.900 |
| `aten::_to_copy` | memory_or_layout | 51 | 1174.000 | 23.020 | 2015.200 |
| `aten::transpose` | memory_or_layout | 80 | 1162.000 | 14.525 | 1256.300 |
| `aten::_to_copy` | memory_or_layout | 51 | 1126.000 | 22.078 | 1394.200 |
| `aten::mm` | matrix_multiply | 16 | 1116.000 | 69.750 | 472.100 |
| `aten::transpose` | memory_or_layout | 80 | 1115.000 | 13.938 | 1259.000 |
| `aten::_to_copy` | memory_or_layout | 51 | 1057.000 | 20.725 | 1464.200 |
| `aten::copy_` | memory_or_layout | 51 | 1023.000 | 20.059 | 750.600 |
| `aten::copy_` | memory_or_layout | 51 | 1017.000 | 19.941 | 733.500 |
| `aten::arange` | other | 40 | 1015.000 | 25.375 | 1318.800 |
| `aten::pow` | other | 17 | 1004.000 | 59.059 | 978.800 |
| `aten::copy_` | memory_or_layout | 16 | 1001.000 | 62.562 | 282.900 |
| `aten::argmax` | other | 5 | 978.000 | 195.600 | 2506.800 |
| `aten::mm` | matrix_multiply | 16 | 962.000 | 60.125 | 493.000 |
| `aten::mm` | matrix_multiply | 16 | 919.000 | 57.438 | 507.800 |
| `aten::copy_` | memory_or_layout | 51 | 891.000 | 17.471 | 670.000 |
| `aten::reshape` | other | 49 | 889.000 | 18.143 | 670.500 |

The classification is a timing-based hypothesis, not a roofline result. The raw Chrome trace is generated locally; operator rows and exact profile dimensions are preserved in the JSON summary.
