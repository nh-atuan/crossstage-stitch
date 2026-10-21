// Forward declaration of cudaStream_t so shared headers compile without the CUDA toolkit.
// Identical to the definition in driver_types.h; a repeated typedef of the same type is legal in C++.
#pragma once

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
