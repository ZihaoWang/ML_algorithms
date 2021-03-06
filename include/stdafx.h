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

using std::size_t;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::numeric_limits;
using std::make_shared;
using std::make_unique;
using std::make_pair;
using std::move;
using std::runtime_error;
using std::logic_error;

typedef arma::vec vec_t;
typedef arma::ivec ivec_t;
typedef arma::rowvec rvec_t;
typedef arma::mat mat_t;
typedef arma::imat imat_t;
typedef arma::cube cube_t;
typedef arma::icube icube_t;
typedef arma::field<mat_t> mf_t;
typedef arma::field<arma::field<mat_t>> f_mf_t;
typedef arma::field<cube_t> cf_t;
typedef arma::field<arma::field<cube_t>> f_cf_t;

#define CRY(expr) \
(throw runtime_error(("\nexception has been caught at file: ") + (string(__FILE__)) + \
               ("\nfunction: ") + (string(__func__)) + \
               ("\nline: ") + (string(std::to_string(__LINE__))) + \
               ("\nwith message: " #expr "\n")))

#endif
