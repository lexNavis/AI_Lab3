#include <armadillo>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void load_csv(arma::mat &X, const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "File not found!\n";
    return;
  }
  /* Here we do not have target */
  std::vector<std::vector<double>> x;
  std::string line;
  /* We do not need header. skip it */
  std::getline(file, line);
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string val;
    std::vector<double> row;
    /* Skip first value in row (Customer ID) we do not need it */
    std::getline(ss, val, ',');
    /* Collect features */
    while (std::getline(ss, val, ',')) {
      row.push_back(std::stod(val));
    }
    /* Save customer's features */
    x.push_back(row);
  }
  /* Transfer data from C++ matrix to arma matrix */
  X.set_size(x.size(), x[0].size());
  for (size_t i = 0; i < x.size(); i++)
    for (size_t j = 0; j < x[i].size(); j++)
      X(i, j) = x[i][j];
}

arma::uvec kmeans(const arma::mat &X, int k, int max_iters, double tol,
                  unsigned seed) {
  /* Find initial clusters' centers
   * Store them in matrix like:
   * center_dimension = centers[cluster][dimension] */
  arma::mat centroids(k, X.n_cols);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, X.n_rows - 1);
  for (int c = 0; c < k; c++) {
    centroids.row(c) = X.row(dist(rng));
  }
  /* Cluster assignment for customers */
  arma::uvec labels(X.n_rows);
  for (int iter = 0; iter < max_iters; iter++) {
    /* Select best cluster for each customer */
    for (size_t i = 0; i < X.n_rows; i++) {
      double min_dist = std::numeric_limits<double>::max();
      int best_cluster = 0;
      for (int j = 0; j < k; j++) {
        /* Best cluster - nearest cluster */
        double d = arma::norm(X.row(i).t() - centroids.row(j).t());
        if (d < min_dist) {
          min_dist = d;
          best_cluster = j;
        }
      }
      /* Assign nearest cluster to customer */
      labels(i) = best_cluster;
    }
    /* When customers are divided, we should
     * find new center of the cluster.
     * New center is average value
     * of its customers' features */
    arma::mat new_centroids(k, X.n_cols, arma::fill::zeros);
    arma::vec counts(k, arma::fill::zeros);
    for (size_t i = 0; i < X.n_rows; i++) {
      int c = labels(i);
      new_centroids.row(c) += X.row(i);
      counts(c) += 1.0;
    }
    /* After accumulating features, find avg */
    for (int c = 0; c < k; c++) {
      if (counts(c) > 0) {
        new_centroids.row(c) /= counts(c);
      } else {
        /* If we have empty cluster, then
         * its center remains the same */
        new_centroids.row(c) = centroids.row(c);
      }
    }
    /* Find difference between old and new center.
     * If diff < tol, exit loop */
    double diff = arma::norm(new_centroids - centroids, "fro");
    /* Save new centroids for next iterations */
    centroids = new_centroids;
    if (diff < tol) {
      break;
    }
  }
  return labels;
}

int main() {
  arma::mat X_base;
  X_base.load("../../../results/part_b/subsample_0_X.csv",
              arma::csv_ascii);
  int k = 2;
  int seed = 42;

  for (int i = 0; i < 5; i++) {
    std::string path =
        "../../../results/part_b/subsample_" + std::to_string(i) + "_X.csv";
    arma::mat X;
    bool ok = X.load(path, arma::csv_ascii);
    if (!ok) {
      std::cerr << "Cannot load " << path << "\n";
      continue;
    }

    std::cout << "Processing subsample " << i << " (" << X.n_rows << " x "
              << X.n_cols << ")\n";

    arma::uvec labels = kmeans(X, k, 300, 1e-4, seed);

    std::string out_path =
        "../../../results/part_b/cpp_labels_" + std::to_string(i) + ".csv";
    labels.save(out_path, arma::csv_ascii);
  }
  return 0;
}