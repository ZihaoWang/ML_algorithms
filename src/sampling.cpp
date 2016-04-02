#include "../include/sampling.h"

//#define UNIT_TESTING

namespace hmc
{

const mat_t *model::run(const std::function<double(vec_t *, const vec_t &, const bool)> &data_func, 
        const int num_sample)
{
    result = make_unique<mat_t>(DIM_INPUT, num_sample, arma::fill::zeros);
    const int num_total = (num_sample + NUM_BURNING) * NUM_SKIP;
    position_old->zeros();
    obj_old = data_func(position_drv.get(), *position, true);

    for (int i = 1; i <= num_total; ++i)
    {
        momentum->randn();
        hamilton_old = 0.5 * arma::accu(momentum->t() * *momentum) + obj_old;

        leapfrog(data_func);
        obj = data_func(position_drv.get(), *position, true);
        hamilton = 0.5 * arma::accu(momentum->t() * *momentum) + obj;

        if (accept())
            update_status();

        if (i % NUM_SKIP == 0)
            if (i / NUM_SKIP > NUM_BURNING)
                result->col(i / NUM_SKIP - NUM_BURNING - 1) = *position_old;
    }

    return result.get();
}

model::model(const init_arg &args) :
    dist(0.0, 1.0)
{
    arma::arma_rng::set_seed_random();
    check_arg(args);
    set_arg(args);

    momentum = make_unique<vec_t>(DIM_INPUT, arma::fill::zeros);
    position_old = make_unique<vec_t>(DIM_INPUT, arma::fill::zeros);
    position = make_unique<vec_t>(DIM_INPUT, arma::fill::zeros);
    position_drv = make_unique<vec_t>(DIM_INPUT, arma::fill::zeros);
    hamilton = 0.0;
    hamilton_old = 0.0;
    obj = 0.0;
    obj_old = 0.0;
}

void model::check_arg(const init_arg &args)
{
    if (args.dim_input == 0)
        CRY();
    if (args.num_burning == 0)
        CRY();
    if (args.num_skip == 0)
        CRY();
    if (args.num_step == 0)
        CRY();
}

void model::set_arg(const init_arg &args)
{
    DIM_INPUT = args.dim_input;
    NUM_BURNING = args.num_burning;
    NUM_SKIP = args.num_skip;
    NUM_STEP = args.num_step;
    STEP_SIZE = args.step_size;
}

} // namespace hmc

#ifdef UNIT_TESTING

double gaussian_func(vec_t *grad, const vec_t &x, const bool need_obj)
{
    double obj = 0.0;
    if (need_obj)
        obj = 0.5 * x[0] * x[0];

    if (grad != nullptr)
        (*grad)[0] = x[0];
    return obj;
}

int main()
{
    hmc::init_arg args;
    args.dim_input = 1;
    args.num_burning = 15;
    hmc::model sampler(args);

    const mat_t *result = sampler.run(gaussian_func, 100);
    cout << "result:\n";
    for (int i = 0; i < result->n_cols; ++i)
        cout << (*result)[i] << ", ";
    cout << endl;

    return 0;
}

#endif
