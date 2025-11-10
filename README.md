# Reallocation of extractors and buffers

## Overview

The typical **tatami** pattern is to create a new `Extractor` and buffer for each pass through the matrix.
For algorithms that need to perform multiple passes through the matrix (e.g., approximate SVD algorithms),
this is theoretically suboptimal as it performs re-allocations that could have been re-used across iterations.
In practice, I doubt this has much impact as the allocation work is small compared to the iteration through the matrix.
Nonetheless, we should probably test it out. 

## Setup

We consider repeated products of a row-major matrix with a double-precision vector.
To recapitulate **tatami**'s behavior, we extract each row into a separate buffer before computing the dot product.
This is repeated for each row to obtain the matrix-vector product, which is eventually summed to obtain a simple summary statistic.
The entire process is then repeated for the desired number of iterations of the cross-product.

In the "reused" approach, the buffer is allocated once outside of the product iterations, and re-used in each iteration.
In the "re-allocated" approach, this buffer is allocated anew within the product iteration loop.
The question is whether the reallocation of the buffer contributes to the total runtime of the entire process.

## Results

No, it doesn't have a significant effect, thank god.
The differences here are minor, sporadic, and sometimes depend on the order in which the methods are run. 

```
$ ./build/multtest 
Testing a 10000 x 2000 matrix

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      287,389,270.00 |                3.48 |    0.5% |      3.17 | `reused`
|      287,582,146.00 |                3.48 |    0.1% |      3.17 | `reallocated`

$ ./build/multtest -r 2000 -c 10000
Testing a 2000 x 10000 matrix

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      295,690,949.00 |                3.38 |    1.1% |      3.24 | `reused`
|      293,861,844.00 |                3.40 |    1.0% |      3.24 | `reallocated`

$ ./build/multtest -r 100000 -c 200
Testing a 100000 x 200 matrix

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      264,561,939.00 |                3.78 |    0.2% |      2.92 | `reused`
|      270,300,380.00 |                3.70 |    0.4% |      2.99 | `reallocated`

$ ./build/multtest -r 200 -c 100000
Testing a 200 x 100000 matrix

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      401,299,832.00 |                2.49 |    0.3% |      4.42 | `reused`
|      401,185,430.00 |                2.49 |    0.1% |      4.42 | `reallocated`
```

Frankly, this is a relief, as we don't have to drag allocations around the place to squeeze out more performance.
Doing so would be an absolute pain for multi-threaded processes where each thread needs its own copy of everything.
A `tatami::OracularExtractor` instance can't be re-used anyway once its predictions have been consumed,
so any re-use strategy would need to switch to `MyopicExtractor`s, which may incur an even greater performance penalty.

## Build instructions

Just use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
