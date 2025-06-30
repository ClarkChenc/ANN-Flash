#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

extern size_t K;         // top-k knns
extern int NUM_THREADS;  // set num_threads
const int REPEATED_COUNT = 1;

const size_t MAX_ITERATIONS = 300;  // k-means iteration times
const size_t SAMPLE_NUM = 200000;   // sample number in generating codebooks
size_t SUBVECTOR_LENGTH = 10;       // PQ subvector length/dimension

const size_t VECTORS_PER_BLOCK = 1;
// const size_t VECTORS_PER_BLOCK = 16;	// calculate multiple neighbor distance at the same time.
// 										// This parameter cannot be
// modified due to the restrict of SIMD shuffle

const size_t M = 32;  // neighbor's number, should be times of 16
const size_t EF_CONSTRUCTION = 400;                      // maximum number of candidate neighbors considered during index construction.
const size_t EF_SEARCH = 10;  // maximum number of candidates retained during the search phase.
const size_t DIRECTION_NUM = 1;

const size_t SUBVECTOR_NUM = 32;  // PQ subvector num, should be times of 16 when using Flash
const size_t CLUSTER_NUM = 256;   // cluster numbers of each subvector

constexpr size_t PRINCIPAL_DIM = 128;  // Rest dimiensions after running PCAdefine USE_PCA
                                       // // use PCA to tallor dimensions
#define USE_PCA
#define USE_PCA_OPTIMAL                // use cumulative variance to group subvectors in PCA

#define SAVE_MEMORY  // not save distance table while using SDC to calculate distance
#define USE_PREFETCH
// #define ALIGN_PREFETCH
#define DEBUG_LOG
// #define TRACE_SEARCH
#define RERANK  // search 2k points to rerank

/* OPTIMIZE OPTIONS for FlashStrategy */
// #define PQLINK_STORE					// save neighbor's vector for each node
// #define PQLINK_CALC						// calculate neighbor data at once

#define PQ_LINK_LRU_SIZE 50000

#define FORBID_RUN

// #define INT8							// data_t type
#define INT16
// #define INT32
#define FLOAT32

// #define RUN_WITH_SSE					// Indicate specific SIMD, SSE can only use INT8
#define RUN_WITH_AVX  // AVX can only use
// #define RUN_WITH_AVX512

#define ALL_POSITIVE_NUMBER  // modify hnswalg to make all operated number positive

#define TAU 8  // parameter used in tau-MG
// #define VBASE						// use vbase to calculate distance
#define VBASE_WINDOW_SIZE 5  // the window size of vbase
// #define ADSAMPLING					// use ADSampling to calculate distance
#define ADSAMPLING_EPSILON 2.1  // the epsilon of ADSampling
#define ADSAMPLING_DELTA_D 4    // the delta_d of ADSampling

#if defined(ALL_POSITIVE_NUMBER)
#if defined(INT8)
typedef uint8_t data_t;
#elif defined(INT16)
typedef uint16_t data_t;
#elif defined(INT32)
typedef uint32_t data_t;
#elif defined(FLOAT32)
typedef float data_t;
#endif
#else
#if defined(INT8)
typedef int8_t data_t;
#elif defined(INT16)
typedef int16_t data_t;
#elif defined(INT32)
typedef int32_t data_t;
#elif defined(FLOAT32)
typedef float data_t;
#endif
#endif

// for flash's dimension rearranging
#if defined(RUN_WITH_AVX512)
#define BATCH 4
#elif defined(RUN_WITH_AVX)
#define BATCH 2
#else
#define BATCH 1
#endif
