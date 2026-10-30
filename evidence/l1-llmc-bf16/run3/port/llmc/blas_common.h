/*
oneMath BLAS related utils (ported from cublas_common.h).
cuBLAS is column-major; we therefore call oneapi::math::blas::column_major
with transposed interpretations of our row-major tensors.
*/
#ifndef BLAS_COMMON_H
#define BLAS_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <sycl/sycl.hpp>
#include <oneapi/math/blas.hpp>

#include "sycl_common.h"

#endif // BLAS_COMMON_H
