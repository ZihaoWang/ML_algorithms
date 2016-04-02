#ifndef EVENSONG_STDAFX
#define EVENSONG_STDAFX

#include <iostream>
#include <fstream>
#include <cstdio>

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <memory>
#include <regex>

#include <limits>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <random>
#include <functional>
#include <type_traits>
#include <stdexcept>

#include <armadillo> // matrix lib

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;
using std::make_unique;
using std::make_pair;
using std::runtime_error;
using std::logic_error;

typedef arma::mat mat_t;
typedef arma::imat imat_t;
typedef arma::vec vec_t;
typedef arma::ivec ivec_t;
typedef arma::rowvec rvec_t;
typedef arma::cube cube_t;
typedef arma::icube icube_t;
typedef arma::field<mat_t> mf_t;
typedef arma::field<arma::field<mat_t>> f_mf_t;

#define CRY(expr) \
(throw runtime_error(("\nexception has been caught at file: ") + (string(__FILE__)) + \
               ("\nfunction: ") + (string(__func__)) + \
               ("\nline: ") + (string(std::to_string(__LINE__))) + \
               ("\nwith message: " #expr "\n")))

inline void check_vector(const vector<int> &vec, const int true_size)
{
    if (vec.size() != true_size)
        CRY();
    else
        for (const auto e : vec)
            if (e <= 0)
                CRY();
}

#endif
