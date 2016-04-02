#include "../include/kernel_function.h"

namespace kernel_func
{

void mixture_ker::comp_ker(mat_t *dst, const mat_t &lhs, const mat_t &rhs)
{
    if (lhs.n_rows != rhs.n_rows)
        CRY("source data(lhs or rhs) of kernel function have different dimensions");
    if (dst->n_rows != lhs.n_cols || dst->n_cols != rhs.n_cols)
        CRY("destination and source have different sizes");

    if (std::abs(a0 - 0.0) > DBL_EPSILON)
    {
        if (std::abs(a1 - 0.0) > DBL_EPSILON)
            for (int i = 0; i < lhs.n_cols; ++i)
                for (int j = 0; j < rhs.n_cols; ++j)
                    (*dst)(i, j) = a0 * std::exp(-0.5 * a1 * arma::norm(lhs.col(i) - rhs.col(j)));
        else
            for (int i = 0; i < lhs.n_cols; ++i)
                for (int j = 0; j < rhs.n_cols; ++j)
                    (*dst)(i, j) = a0;
    }
    else
        dst->zeros();
    if (std::abs(a2 - 0.0) > DBL_EPSILON)
    {
        for (int i = 0; i < lhs.n_cols; ++i)
            for (int j = 0; j < rhs.n_cols; ++j)
                (*dst)(i, j) += a2 * arma::dot(lhs.col(i), rhs.col(j));
    }
    if (std::abs(a3 - 0.0) > DBL_EPSILON)
        *dst += a3;
}

}
