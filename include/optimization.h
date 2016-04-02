#ifndef EVENSONG_OPTIMIZATION
#define EVENSONG_OPTIMIZATION

#include "./stdafx.h"
#include "./linear_search.h"
#include "./field_algorithm.h"

//#define DEBUG

namespace optimization
{

template <typename VAL_T, typename METHOD_T>
class wrapper final
{
    public :
        template <typename INIT_ARG_T>
        wrapper(const INIT_ARG_T &args, const VAL_T &val)
        {
            done_bfr_func = false;
            optimizer = make_unique<METHOD_T>(args, val);
        }
        
        void optimize(VAL_T *val, 
                const std::function<void(const VAL_T &)> &set_ptr_func,
                const std::function<double(VAL_T *)> &data_func,
                const std::function<void(const int)> &extra_func = std::function<void(const int)>(),
                const std::function<void()> &after_update_func = std::function<void()>())
        {
            if (!done_bfr_func && optimizer->is_single_optimization())
            {
                optimizer->before_optimize(val, set_ptr_func, data_func, after_update_func);
                done_bfr_func = true;
            }
            else
                optimizer->before_optimize(val, set_ptr_func, data_func, after_update_func);

            optimizer->optimize(val, data_func, extra_func, after_update_func);
            if (!optimizer->is_single_optimization())
                optimizer->after_optimize(*val, set_ptr_func);
        }

        bool has_converged() const 
        {
            return optimizer->has_converged();
        }

        int get_epoch() const 
        {
            return optimizer->get_epoch() + 1;
        }

        void alter_rate() const
        {
            optimizer->alter_rate();
        }

    private :
        bool done_bfr_func;
        unique_ptr<METHOD_T> optimizer;
};

namespace grad_dsct
{

using namespace field_algorithm;

struct init_arg
{
    int batch_size = 0;
    const double init_learning_rate = 0.005;
};

template <typename VAL_T>
class optimizer final
{
    public :
        optimizer(const init_arg &args, const VAL_T &val)
        {
            if (args.batch_size <= 0)
                CRY();
            batch_size = args.batch_size;
            init_learning_rate = args.init_learning_rate;
            rate_now = init_learning_rate;
            converged = false;
            mf_init(&new_grad, val);
        }

        void before_optimize(VAL_T *val, 
                const std::function<void(const VAL_T &)> &set_ptr_func,
                const std::function<double(VAL_T *)> &data_func,
                const std::function<void()> &after_update_func)
        {}

        void optimize(VAL_T *val, const std::function<double(VAL_T *)> &data_func, 
                const std::function<void(int)> &extra_func, const std::function<void()> &after_update_func)
        {
            data_func(&new_grad);
            //cout << new_grad(0,0)(0,0) << endl;
            mf_for_each(val, new_grad, [this](mat_t &dst, const mat_t &src){ dst -= rate_now * src; });
        }

        void after_optimize(const VAL_T &val, const std::function<void(const VAL_T &)> &set_ptr_func) {}

        bool is_single_optimization() { return true; }

        void reset_optm()
        {
            rate_now = init_learning_rate;
            converged = false;
        }

        bool has_converged()
        {
            return converged;
        }

        void alter_rate()
        {
            if (rate_now * 0.8 > 0.00001)
                rate_now *= 0.8;
            else
            {
                rate_now = 0.00001;
                converged = true;
            }
        }

        void name() { cout << "grad dsct" << endl; }

    private :

        VAL_T new_grad;
        int idx_now;
        int batch_size;
        double init_learning_rate;
        double rate_now;
        bool converged;
};

} // namespace grad_dsct

namespace l_bfgs
{

using namespace field_algorithm;

struct init_arg final
{
    const double c1 = 0.0001;
    const double c2 = 0.9; // 0.9 for Newton or quasi-Newton method
    const int max_num_history = 10;
    int max_epoch = 0;
};

template <typename VAL_T>
class optimizer final
{
    public :
        optimizer(const init_arg &args, const VAL_T &val);

        void before_optimize(VAL_T *val,
                const std::function<void(const VAL_T &)> &set_ptr_func,
                const std::function<double(VAL_T *)> &data_func,
                const std::function<void()> &after_update_func)
        {
            new_obj = data_func(&new_grad);
            wolfe->set_new_val(*val);
            set_ptr_func(wolfe->get_new_val_ref());
            if (after_update_func)
                after_update_func();

            old_val = *val;
            old_obj = new_obj;
            old_grad = new_grad;
        }

        void optimize(VAL_T *val, const std::function<double(VAL_T *)> &data_func, 
                const std::function<void(const int)> &extra_func, const std::function<void()> &after_update_func)
        {
            if (has_converged())
                return;
            if (is_single_optimization())
                single_optimize(val, data_func, extra_func, after_update_func);
            else
                multi_optimize(val, data_func, extra_func, after_update_func);
        }

        void after_optimize(const VAL_T &val,
                const std::function<void(const VAL_T &)> &set_ptr_func) 
        { set_ptr_func(val); }

        bool is_single_optimization() { return (max_epoch == 1 ? true : false); }

        bool has_converged() const { return converged; }

        int get_epoch() const { return epoch; }

        void alter_rate() const { throw logic_error("l_bfgs cannot be altered rate by user"); }

        void name() { cout << "l-bfgs" << endl; }

    private :

        void single_optimize(VAL_T *val, const std::function<double(VAL_T *)> &data_func, 
                const std::function<void(const int)> &extra_func, const std::function<void()> &after_update_func)
        {
            comp_new_val(val, data_func, extra_func, after_update_func);
            check_convergence(*val);
            update_member(*val);
            ++epoch;
        }

        void multi_optimize(VAL_T *val, const std::function<double(VAL_T *)> &data_func, 
                const std::function<void(const int)> &extra_func, const std::function<void()> &after_update_func)
        {
            for (; epoch < max_epoch; ++epoch)
            {
                comp_new_val(val, data_func, extra_func, after_update_func);
                check_convergence(*val);
                if (has_converged())
                    break;
                update_member(*val);
            }
        }

        void comp_new_val(VAL_T *val, const std::function<double(VAL_T *)> &data_func, 
        const std::function<void(const int)> &extra_func, const std::function<void()> &after_update_func);

        void check_convergence(const VAL_T &new_val);

        void update_member(const VAL_T &new_val);

        void get_hessian_inv();

        void get_search_dir();

        bool grad_norm_is_small() { return mf_norm2(new_grad) < 1e-6 ? true : false; }

        void init_wolfe(const init_arg &args, const VAL_T &val);

        void check_args(const init_arg &args);

        int epoch;
        int max_epoch;
        bool converged;
        int max_num_history;
        linear_search::wolfe_rule::init_arg wolfe_args;
        unique_ptr<linear_search::wolfe_rule::searcher<VAL_T>> wolfe;

        VAL_T old_val;
        double old_obj;
        double new_obj;
        VAL_T old_grad;
        VAL_T new_grad;

        vector<VAL_T> s;
        vector<VAL_T> y;
        vector<double> alpha;
        vector<double> rho;
        VAL_T search_dir;
        double hessian_inv; // a diagonal matrix with identity diagonal values, so we represent it as a single double-value
};

template <typename VAL_T>
void optimizer<VAL_T>::comp_new_val(VAL_T *val, const std::function<double(VAL_T *)> &data_func, 
        const std::function<void(const int)> &extra_func, const std::function<void()> &after_update_func)
{
    get_hessian_inv();
    get_search_dir();
    if (!wolfe->run(val, &new_obj, &new_grad, search_dir, data_func, after_update_func))
        throw runtime_error("wolfe linear search failed");

    if (extra_func)
        extra_func(epoch);
}

template <typename VAL_T>
void optimizer<VAL_T>::check_convergence(const VAL_T &new_val)
{        
    if (new_val == old_val)
    {
        cout << "optimization finishes, exit status: new value == old value" << endl;
        converged = true;
    }

    const double stable_obj = std::max(1.0, 
            std::max(std::abs(new_obj), std::abs(old_obj)));
    if ((old_obj - new_obj) / stable_obj < 1e-15)
    {
        cout << "optimization finishes, exit status: new objective value == old objective value" << endl;
        converged = true;
    }

    if (grad_norm_is_small() && epoch > 0)
    {
        cout << "optimization finishes, exit status: gradient norm is small" << endl;
        converged = true;
    }
}

template <typename VAL_T>
void optimizer<VAL_T>::update_member(const VAL_T &new_val)
{
    const int pos = epoch % max_num_history;
    mf_for_each(&s[pos], new_val, old_val, [](mat_t &dst, const mat_t &src1, const mat_t &src2)
            { dst = src1 - src2; });
    mf_for_each(&y[pos], new_grad, old_grad, [](mat_t &dst, const mat_t &src1, const mat_t &src2)
            { dst = src1 - src2; });
    old_val = new_val;
    old_obj = new_obj;
    old_grad = new_grad;
}

template <typename VAL_T>
void optimizer<VAL_T>::get_hessian_inv()
{
    if (epoch > 0)
    {
        int prv_pos = (epoch - 1) % max_num_history;
        hessian_inv = mf_dot(s[prv_pos], y[prv_pos]) / mf_dot(y[prv_pos], y[prv_pos]);
    }
    else
        hessian_inv = 1.0 / std::sqrt(mf_dot(new_grad, new_grad));
}

template <typename VAL_T>
void optimizer<VAL_T>::get_search_dir()
{
    search_dir = new_grad;
    int limit = (max_num_history > epoch) ? 0 : (epoch - max_num_history);

    for (int i = epoch; i > limit; --i)
    {
        const int pos = (i + max_num_history - 1) % max_num_history;
        rho[epoch - i] = 1.0 / mf_dot(y[pos], s[pos]);
        const double tmp = rho[epoch - i] * mf_dot(s[pos], search_dir);
        alpha[epoch - i] = tmp;
        mf_for_each(&search_dir, y[pos], 
                [tmp](mat_t &dst, const mat_t &src1)
                { dst -= tmp * src1; });
    }

    mf_for_each(&search_dir, [this](mat_t &dst){ dst *= hessian_inv; });
    for (int i = limit; i < epoch; ++i)
    {
        const int pos = i % max_num_history;
        const double beta = rho[epoch - i - 1] * mf_dot(y[pos], search_dir);
        const double tmp = alpha[epoch - i - 1] - beta;
        mf_for_each(&search_dir, s[pos], 
                [tmp](mat_t &dst, const mat_t &src1)
                { dst += tmp * src1; }); 
    }
    mf_for_each(&search_dir, [](mat_t &dst){ dst *= -1; });
}

template <typename VAL_T>
optimizer<VAL_T>::optimizer(const init_arg &args, const VAL_T &val)
{
    check_args(args);
    converged = false;
    max_num_history = args.max_num_history;
    max_epoch = args.max_epoch;
    hessian_inv = 1.0;
    epoch = 0;
    init_wolfe(args, val);

    s.reserve(max_num_history);
    y.reserve(max_num_history);
    alpha.reserve(max_num_history);
    rho.reserve(max_num_history);
    for (int i = 0; i < max_num_history; ++i)
    {
        s.emplace_back();
        y.emplace_back();
        mf_init(&s.back(), val);
        mf_init(&y.back(), val);
        alpha.emplace_back(0.0);
        rho.emplace_back(0.0);
    }
    mf_init(&old_val, val);
    old_obj = 0.0;
    new_obj = 0.0;
    mf_init(&old_grad, val);
    mf_init(&new_grad, val);
    mf_init(&search_dir, val);

}

template <typename VAL_T>
void optimizer<VAL_T>::init_wolfe(const init_arg &args, const VAL_T &val)
{
    wolfe_args.c1 = args.c1;
    wolfe_args.c2 = args.c2;
    wolfe = make_unique<linear_search::wolfe_rule::searcher<VAL_T>>(wolfe_args, val);
}

template <typename VAL_T>
void optimizer<VAL_T>::check_args(const init_arg &args)
{
    static_assert((std::is_same<VAL_T, mf_t>::value ||
                std::is_same<VAL_T, f_mf_t>::value), 
            "\ntype VAL_T of l_bfgs::optimizer must be mf_t or f_mf_t");

    if (args.max_epoch == 0)
        CRY("max_epoch == 0");
}


} // namespace l_bfgs

} // namespace optimization
#endif
