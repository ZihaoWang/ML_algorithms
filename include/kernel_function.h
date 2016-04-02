#ifndef EVENSONG_KERNEL_FUNCTION
#define EVENSONG_KERNEL_FUNCTION

#include "./stdafx.h"
#include "./prepare_data.h"

namespace kernel_func
{

/*
 * we can use virtual function here since computation of kernel matrix is not called frequently:
 * it is called once for all the data.
 */

class ker_base
{
    public :
        ker_base() {}
        virtual ~ker_base() {};
        virtual void comp_ker(mat_t *dst, const mat_t &lhs, const mat_t &rhs) = 0;
};

/*
 * k(x, y) = a0 * exp(-0.5 * a1 * ||x - y||^2) + a2 * x * y + a3
 */
class mixture_ker final : public ker_base
{
    public :
        mixture_ker() :
            ker_base()
        {
            std::default_random_engine gen;
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            a0 = dist(gen);
            a1 = dist(gen);
            a2 = dist(gen);
            a3 = dist(gen);
        }

        mixture_ker(const double _a0, const double _a1, const double _a2, const double _a3) :
            ker_base()
        {
            set_arg(_a0, _a1, _a2, _a3);
        }

        ~mixture_ker() override {}

        void comp_ker(mat_t *dst, const mat_t &lhs, const mat_t &rhs) override;

        void set_arg(const double _a0, const double _a1, const double _a2, const double _a3)
        {
            a0 = _a0;
            a1 = _a1;
            a2 = _a2;
            a3 = _a3;
        }

    private :
        double a0;
        double a1;
        double a2;
        double a3;
};

}

#endif

