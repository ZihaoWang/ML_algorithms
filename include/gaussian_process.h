#ifndef EVENSONG_GAUSSIAN_PROCESS
#define EVENSONG_GAUSSIAN_PROCESS

#include "./stdafx.h"
#include "./kernel_function.h"

namespace gp
{

struct init_arg
{
    int num_training_data = 0;
    int num_testing_data = 0;
    double var_noise = 0.0;
    std::function<void(vec_t *mean, const mat_t &input)> mean_func;
    unique_ptr<kernel_func::ker_base> cov_func;
};

class model
{
    public :

        model(init_arg *args);

        void train(const mat_t &input, const vec_t &target);

        pair<const vec_t *, const vec_t *> regress(const mat_t &training_input, const mat_t &testing_input);

    private :

        void check_arg(const init_arg &args)
        {
            if (args.num_training_data <= 0)
                CRY();
            if (args.num_testing_data <= 0)
                CRY();
            if (std::abs(args.var_noise - 0.0) < DBL_EPSILON)
                cout << "gaussian process has no noise" << endl;
            if (!args.mean_func)
                cout << "gaussian process has no mean function, use 0 by default" << endl;
            if (args.cov_func == nullptr)
                CRY();
        }

        void set_arg(init_arg *args)
        {
            NUM_TRAINING_DATA = args->num_training_data;
            NUM_TESTING_DATA = args->num_testing_data;
            VAR_NOISE = args->var_noise;
            MEAN_FUNC = args->mean_func;
            COV_FUNC = std::move(args->cov_func);
        }

        int NUM_TRAINING_DATA;
        int NUM_TESTING_DATA;
        int VAR_NOISE;
        std::function<void(vec_t *mean, const mat_t &input)> MEAN_FUNC;
        unique_ptr<kernel_func::ker_base> COV_FUNC;
        
        unique_ptr<vec_t> training_mean;
        unique_ptr<mat_t> training_cov;
        unique_ptr<mat_t> chol_cov;
        unique_ptr<vec_t> mixture_cov;
        unique_ptr<vec_t> testing_cov;
        unique_ptr<vec_t> alpha;
        unique_ptr<vec_t> v;
        unique_ptr<vec_t> testing_target;
        unique_ptr<vec_t> testing_var;
};

} // namespace gp

#endif
