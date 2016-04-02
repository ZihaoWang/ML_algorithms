#ifndef ANDY_ACTIVATION_FUNCTION
#define ANDY_ACTIVATION_FUNCTION

#include "./stdafx.h"

namespace act_func
{

template <typename FUNC_T>
class wrapper
{
    public :
        double obj(const double x) const { return activation_func.obj(x); }
        double drv(const double x) const { return activation_func.drv(x); }
        double drv_wrt_result(const double x) const { return activation_func.drv_wrt_result(x); }

    private :
        FUNC_T activation_func;
};

class hyp_tan final
{
    public :
        double obj(const double x) const
        {
            return std::tanh(x);
        }

        double drv(const double x) const
        {
            double tmp = std::tanh(x);
            return (1.0 + tmp) * (1.0 - tmp);
        }

        double drv_wrt_result(const double x) const
        {
            return (1.0 + x) * (1.0 - x);
        }
};

class sigm final
{
    public :
        double obj(const double x) const
        {
            return 1.0 / (1.0 + std::exp(-x));
        }

        double drv(const double x) const
        {
            return obj(x) * (1.0 - obj(x));
        }

        double drv_wrt_result(const double x) const
        {
            return x * (1.0 - x);
        }
};

class re_lu final
{
    public :
        double obj(const double x) const
        {
            return std::max(0.0, x);
        }

        double drv(const double x) const
        {
            return x > 0.0 ? 1.0 : 0.0;
        }

        double drv_wrt_result(const double x) const
        {
            return drv(x);
        }
};

}

#endif

