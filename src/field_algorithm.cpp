#include "../include/field_algorithm.h"

namespace field_algorithm
{

void mf_init(mf_t *dst, const mf_t &src)
{
    dst->copy_size(src);
    auto x_it = dst->begin();
    auto y_it = src.begin();
    for (; x_it != dst->end(); ++x_it, ++y_it)
        *x_it = mat_t((*y_it).n_rows, (*y_it).n_cols, arma::fill::zeros);
    if (y_it != src.end())
        CRY("field sizes don't match");
}

void mf_init(f_mf_t *dst, const f_mf_t &src)
{
    dst->copy_size(src);
    auto x_it = dst->begin();
    auto y_it = src.begin();
    for(; x_it != dst->end(); ++x_it, ++y_it)
        mf_init(&*x_it, *y_it);
    if (y_it != src.end())
        CRY("field sizes don't match");
}

void m_for_each(mat_t *dst, mat_t *src, const function<void(double &, double &)> &func)
{
    auto d_it = dst->begin();
    auto s_it = src->begin();
    for (; d_it != dst->end(); ++d_it, ++s_it)
        func(*d_it, *s_it);
    if (s_it != src->end())
        CRY("field sizes don't match");
}

void mf_for_each(mf_t *dst, const mf_t &src1, const function<void(mat_t &, const mat_t &)> &func)
{
    auto d_it = dst->begin();
    auto s1_it = src1.begin();
    for (; d_it != dst->end(); ++d_it, ++s1_it)
        func(*d_it, *s1_it);
    if (s1_it != src1.end())
        CRY("field sizes don't match");
}

void mf_for_each(mf_t *dst, mf_t *src1, const function<void(mat_t &, mat_t &)> &func)
{
    auto d_it = dst->begin();
    auto s1_it = src1->begin();
    for (; d_it != dst->end(); ++d_it, ++s1_it)
        func(*d_it, *s1_it);
    if (s1_it != src1->end())
        CRY("field sizes don't match");
}

void mf_for_each(mf_t *dst, const mf_t &src1, const mf_t &src2, const function<void(mat_t &, const mat_t &, const mat_t &)> &func)
{
    auto d_it = dst->begin();
    auto s1_it = src1.begin();
    auto s2_it = src2.begin();
    for (; d_it != dst->end(); ++d_it, ++s1_it, ++s2_it)
        func(*d_it, *s1_it, *s2_it);
    if (s1_it != src1.end() || s2_it != src2.end())
        CRY("field sizes don't match");
}

void mf_for_each(f_mf_t *dst, const f_mf_t &src1, const function<void(mat_t &, const mat_t &)> &func)
{
    auto d_it = dst->begin();
    auto s1_it = src1.begin();
    for (; d_it != dst->end(); ++d_it, ++s1_it)
        mf_for_each(&*d_it, *s1_it, func);
    if (s1_it != src1.end())
        CRY("field sizes don't match");
}

void mf_for_each(f_mf_t *dst, const f_mf_t &src1, const f_mf_t &src2, const function<void(mat_t &, const mat_t &, const mat_t &)> &func)
{
    auto d_it = dst->begin();
    auto s1_it = src1.begin();
    auto s2_it = src2.begin();
    for (; d_it != dst->end(); ++d_it, ++s1_it, ++s2_it)
        mf_for_each(&*d_it, *s1_it, *s2_it, func);
    if (s1_it != src1.end() || s2_it != src2.end())
        CRY("field sizes don't match");
}

double mf_compute(const mf_t &src1, const mf_t &src2, const double init,
        const function<double(const mat_t &, const mat_t &)> &func1,
        const function<void(double &, const double)> &func2)
{
    auto s1_it = src1.begin();
    auto s2_it = src2.begin();
    double result = init;
    for (; s1_it != src1.end(); ++s1_it, ++s2_it)
        func2(result, func1(*s1_it, *s2_it));

    if (s2_it != src2.end())
        CRY("field sizes don't match");
    return result;
}

double mf_dot(const f_mf_t &src1, const f_mf_t &src2)
{
    auto s1_it = src1.begin();
    auto s2_it = src2.begin();
    double result = 0.0;
    for (; s1_it != src1.end(); ++s1_it, ++s2_it)
            result += mf_compute(*s1_it, *s2_it, 0.0, 
                    [](const mat_t &lhs, const mat_t &rhs){ return arma::accu(lhs % rhs); },
                    [](double &lhs, const double rhs){ lhs += rhs; });

    if (s2_it != src2.end())
        CRY("field sizes don't match");
    return result;
}

bool operator==(const f_mf_t &lhs, const f_mf_t &rhs)
{
    auto l_it = lhs.begin();
    auto r_it = rhs.begin();
    for (; l_it != lhs.end(); ++l_it, ++r_it)
        if (*l_it != *r_it)
            return false;
    if (r_it != rhs.end())
        CRY("field sizes don't match");
    return true;
}

} // namespace field_algorithm
