#include "util.h"
#include "third_party/eigen/Eigen/Dense"
#include "include/strategy_include.h"
#include "include/core.h"

#include <gperftools/heap-profiler.h>

size_t K;
int NUM_THREADS;

int main(int argc, char** argv) {
  std::string dataset = argv[1];
  std::string solve_strategy = argv[2];
  NUM_THREADS = std::stoi(argv[3]);
  K = std::stoi(argv[4]);

  std::cout << "thread_num: " << NUM_THREADS << std::endl;
  std::cout << "topk: " << K << std::endl;

  // Initialization
  std::string source_path;
  std::string query_path;
  std::string gt_path;
  std::string knn_path;
  std::string codebooks_path;
  std::string index_path;

  // Create a filename for saving the index
  std::string suffix = solve_strategy + "_";

  suffix += std::to_string(EF_CONSTRUCTION) + "_";
  suffix += std::to_string(M);

  if (solve_strategy == "flash") {
    suffix += "_";
#if defined(INT8)
    suffix += "INT8_";
#elif defined(INT16)
    suffix += "INT16_";
#elif defined(INT32)
    suffix += "INT32_";
#elif defined(FLOAT32)
    suffix += "FLOAT32_";
#endif
    suffix += std::to_string(SUBVECTOR_NUM) + "_";
    suffix += std::to_string(CLUSTER_NUM) + "_";

#if defined(USE_PCA)
    suffix += "PCA_";
#else
    suffix += "NOPCA_";
#endif
    suffix += std::to_string(PRINCIPAL_DIM);
  } else if (solve_strategy == "flash-v2") {
    suffix += "_";
    suffix += std::to_string(SUBVECTOR_NUM) + "_";
    suffix += std::to_string(CLUSTER_NUM);
  } else if (solve_strategy == "flash-v4" || solve_strategy == "flash-v5") {
    suffix += "_";
    suffix += std::to_string(SUBVECTOR_NUM) + "_";
    suffix += std::to_string(CLUSTER_NUM);
  } else if (solve_strategy == "hnsw-v2") {
    suffix += "_";
    suffix += std::to_string(DIRECTION_NUM);
  }

#if defined(L2)
  suffix += "_L2";
#else
  suffix += "_IP";
#endif

  suffix += ".txt";

  source_path = "../data/" + dataset + "/" + dataset + "_base.fvecs";
  query_path = "../data/" + dataset + "/" + dataset + "_query.fvecs";
  gt_path = "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
  knn_path = "../statistics/knns/" + dataset + "_knn.ivecs";
  codebooks_path = "../statistics/codebooks/" + dataset + "/codebooks_" + suffix;
  index_path = "../statistics/codebooks/" + dataset + "/index_" + suffix;

  SolveStrategy* strategy;
  if (solve_strategy == "hnsw") {
    strategy = new HnswStrategy(source_path, query_path, codebooks_path, index_path);
  } else if (solve_strategy == "hnsw-v2") {
    strategy = new HnswStrategy_V2(source_path, query_path, codebooks_path, index_path);
  } else if (solve_strategy == "flash") {
    strategy = new FlashStrategy(source_path, query_path, codebooks_path, index_path);
  } else if (solve_strategy == "flash-v4") {
    strategy = new FlashStrategy_V4(source_path, query_path, codebooks_path, index_path);
  } else if (solve_strategy == "flash-v5") {
    strategy = new FlashStrategy_V5(source_path, query_path, codebooks_path, index_path);
  } else {
    std::cout << "Unknown strategy: " << solve_strategy << std::endl;
    std::cout << "['hnsw', 'hnsw-v2', 'flash', 'flash-v4']" << std::endl;
    return 1;
  }

  // HeapProfilerStart("heap_profile");

  // Processing
  strategy->solve();
  strategy->save_knn(knn_path);
  strategy->recall(gt_path);

  // HeapProfilerDump("done");
  // HeapProfilerStop();

  return 0;
}
