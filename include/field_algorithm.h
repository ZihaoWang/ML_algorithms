#ifndef EVENSONG_FIELD_ALGORITHM
#define EVENSONG_FIELD_ALGORITHM

#include "./stdafx.h"

namespace field_algorithm
{

using std::function;

void mf_init(mf_t *dst, const mf_t &src);

void mf_init(f_mf_t *dst, const f_mf_t &src);


inline void mf_fill(mf_t *dst, double d)
{
    for (auto it = dst->begin(); it != dst->end(); ++it)
        (*it).fill(d); // in field<field<mat>>, outer iterator doesn't overload operator ->
}

inline void mf_fill(f_mf_t *dst, double d)
{
    for (auto it = dst->begin(); it != dst->end(); ++it)
        mf_fill(&*it, d);
}

void m_for_each(mat_t *dst, mat_t *src, const function<void(double &, double &)> &func);

inline void mf_for_each(mf_t *dst, const function<void(mat_t &)> &func) { std::for_each(dst->begin(), dst->end(), func); }

void mf_for_each(mf_t *dst, const mf_t &src1, const function<void(mat_t &, const mat_t &)> &func);

void mf_for_each(mf_t *dst, mf_t *src1, const function<void(mat_t &, mat_t &)> &func);

void mf_for_each(mf_t *dst, const mf_t &src1, const mf_t &src2, const function<void(mat_t &, const mat_t &, const mat_t &)> &func);

inline void mf_for_each(f_mf_t *dst, const function<void(mat_t &)> &func)
{
    for (auto it = dst->begin(); it != dst->end(); ++it)
        mf_for_each(&*it, func);
}

void mf_for_each(f_mf_t *dst, const f_mf_t &src1, const function<void(mat_t &, const mat_t &)> &func);

void mf_for_each(f_mf_t *dst, const f_mf_t &src1, const f_mf_t &src2, const function<void(mat_t &, const mat_t &, const mat_t &)> &func);


double mf_compute(const mf_t &src1, const mf_t &src2, const double init,
        const function<double(const mat_t &, const mat_t &)> &func1,
        const function<void(double &, const double)> &func2);


inline double mf_dot(const mf_t &src1, const mf_t &src2)
{
    return mf_compute(src1, src2, 0.0, 
            [](const mat_t &lhs, const mat_t &rhs){ return arma::accu(lhs % rhs); },
            [](double &lhs, const double rhs){ lhs += rhs; });
}

double mf_dot(const f_mf_t &src1, const f_mf_t &src2);


template <typename VAL_T>
inline double mf_norm2(const VAL_T &val)
{
    return std::sqrt(mf_dot(val, val));
}

inline bool operator==(const mf_t &lhs, const mf_t &rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](const mat_t &lhs, const mat_t &rhs){ return (arma::accu(lhs != rhs) == 0) ? true : false; });
}

inline bool operator!=(const mf_t &lhs, const mf_t &rhs) { return !(lhs == rhs); }

bool operator==(const f_mf_t &lhs, const f_mf_t &rhs);

inline bool operator!=(const f_mf_t &lhs, const f_mf_t &rhs) { return !(lhs == rhs); }


} // namespace field_transform

#endif
