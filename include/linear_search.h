#ifndef EVENSONG_LINEAR_SEARCH
#define EVENSONG_LINEAR_SEARCH

#include "./stdafx.h"
#include "./field_algorithm.h"

//#define DEBUG

namespace linear_search
{

namespace wolfe_rule
{

using namespace field_algorithm;

struct init_arg final
{
    double c1 = 1e-4;
    double c2 = 0.0; // 0.9 for Newton or quasi-Newton method; 0.1 for conjugate gradient method
};

template <typename VAL_T>
class searcher final
{

    public :
        searcher(const init_arg &args, const VAL_T &val)
        {
            c1 = args.c1;
            c2 = args.c2;
            mf_init(&new_val, val);
        }

        bool run(VAL_T *val, double *obj, VAL_T *grad, const VAL_T &dir, 
                const std::function<double(VAL_T *)> &data_func,
                const std::function<void()> &after_update_func,
                const double alpha_init = 1.0, const int max_iter = 100);

        const VAL_T &get_new_val_ref() const { return new_val; }

        void set_new_val(const VAL_T &val) { new_val = val; }

    private :
        VAL_T new_val;
        double c1;
        double c2; 
        const double min_alpha = 1e-20;
        const double max_alpha = 1e20;
};

template <typename VAL_T>
bool searcher<VAL_T>::run(VAL_T *val, double *obj, VAL_T *grad, const VAL_T &dir, 
                const std::function<double(VAL_T *)> &data_func,
                const std::function<void()> &after_update_func,
                const double alpha_init, const int max_iter)
{
    double phi0_dash = mf_dot(dir, *grad);
    double alpha = alpha_init;
    double old_obj = *obj;

    if (phi0_dash > 0.0)
        return false;

    //cout << "start linear search\n";
    for (int i = 0; i < max_iter && alpha > min_alpha && alpha < max_alpha; ++i)
    {
        //cout << "\tepoch: " << i + 1 << endl;
        mf_for_each(&new_val, *val, dir, [alpha](mat_t &dst, const mat_t &src1, const mat_t &src2){ dst = src1 + alpha * src2; });
        if (after_update_func)
            after_update_func();
        *obj = data_func(grad);

        if (*obj > old_obj + c1 * alpha * phi0_dash)
            alpha *= 0.5;
        else
        {
            const double phi_dash = mf_dot(dir, *grad);

            // strong wolfe condition is applied
            if (phi_dash < c2 * phi0_dash) 
                alpha *= 2.1;
            else
            {
                if (phi_dash > -c2 * phi0_dash)
                    alpha *= 0.5;
                else
                    break;
            }
        }   
    }
    //cout << "final alpha: " << alpha << endl;
    *val = new_val;
    return true;
}

} // namespace wolfe_rule

} // namespace linear_search

#endif
