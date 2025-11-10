#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <vector>

#include "tatami/tatami.hpp"

template<bool reallocate_>
double multiply(const tatami::Matrix<double, int>& matrix, const int iterations) {
    auto NR = matrix.nrow();
    auto NC = matrix.ncol();

    std::mt19937_64 rng(123456);
    std::normal_distribution<double> ndist;
    std::vector<double> rhs(NC);

    auto outer_buffer = [&]{
        if constexpr(reallocate_) {
            return true;
        } else {
            return std::vector<double>(NC);
        }
    }();
    auto outer_ext = [&]{
        if constexpr(reallocate_) {
            return true;
        } else {
            return matrix.dense_row();
        }
    }();

    double output = 0;
    for (int it = 0; it < iterations; ++it) {
        auto inner_buffer = [&]{
            if constexpr(reallocate_) {
                return std::vector<double>(NC);
            } else {
                return true;
            }
        }();
        auto inner_ext = [&]{
            if constexpr(reallocate_) {
                return matrix.dense_row();
            } else {
                return true;
            }
        }();

        auto& buffer = [&]() -> std::vector<double>& {
            if constexpr(reallocate_) {
                return inner_buffer;
            } else {
                return outer_buffer;
            }
        }();
        auto& ext = [&]() -> std::unique_ptr<tatami::MyopicDenseExtractor<double, int> >& {
            if constexpr(reallocate_) {
                return inner_ext;
            } else {
                return outer_ext;
            }
        }();

        for (auto& s : rhs) {
            s = ndist(rng);
        }

        double sum = 0;
        for (decltype(NR) r = 0; r < NR; ++r) {
            const auto ptr = ext->fetch(r, buffer.data());
            tatami::copy_n(ptr, NC, buffer.data());
            sum += std::inner_product(buffer.begin(), buffer.end(), rhs.data(), 0.0);
        }

        output += sum;
    }

    return output;
}

int main(int argc, char* argv[]) {
    CLI::App app{"Expanded testing checks"};
    int nr;
    app.add_option("-r,--nrow", nr, "Number of rows")->default_val(10000);
    int nc;
    app.add_option("-c,--ncol", nc, "Number of columns")->default_val(2000);
    int iterations;
    app.add_option("-i,--iter", iterations, "Number of iterations")->default_val(10);
    CLI11_PARSE(app, argc, argv);

    std::cout << "Testing a " << nr << " x " << nc << " matrix" << std::endl;
    std::mt19937_64 rng(98765);
    std::vector<double> contents(nr * nc);
    std::normal_distribution<double> ndist;
    for (auto& x : contents) {
        x = ndist(rng);
    }
    tatami::DenseMatrix<double, int, std::vector<double> > matrix(nr, nc, std::move(contents), true);

    double expected = -1;
    ankerl::nanobench::Bench().run("reallocated", [&](){
        double sum = multiply<true>(matrix, iterations);
        if (expected < 0) {
            expected = sum;
        } else if (std::abs(sum - expected) > 0.00001) {
            std::cerr << "WARNING: different result from outer summation (" << sum << ")" << std::endl;
        }
    });


    ankerl::nanobench::Bench().run("reused", [&](){
        double sum = multiply<false>(matrix, iterations);
        if (expected < 0) {
            expected = sum;
        } else if (std::abs(sum - expected) > 0.00001) {
            std::cerr << "WARNING: different result from outer summation (" << sum << ")" << std::endl;
        }
    });

    return 0;
}
