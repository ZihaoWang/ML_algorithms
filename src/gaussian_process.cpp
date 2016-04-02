#include "../include/gaussian_process.h"

#define UNIT_TESTING

namespace gp
{

model::model(init_arg *args)
{
    arma::arma_rng::set_seed_random();
    check_arg(*args);
    set_arg(args);

    training_mean = make_unique<vec_t>(NUM_TRAINING_DATA, arma::fill::zeros);
    training_cov = make_unique<mat_t>(NUM_TRAINING_DATA, NUM_TRAINING_DATA, arma::fill::zeros);
    chol_cov = make_unique<mat_t>(NUM_TRAINING_DATA, NUM_TRAINING_DATA, arma::fill::zeros);
    mixture_cov = make_unique<vec_t>(NUM_TRAINING_DATA, arma::fill::zeros);
    testing_cov = make_unique<vec_t>(1, arma::fill::zeros);
    alpha = make_unique<vec_t>(NUM_TRAINING_DATA, arma::fill::zeros);
    v = make_unique<vec_t>(NUM_TRAINING_DATA, arma::fill::zeros);
    testing_target = make_unique<vec_t>(NUM_TESTING_DATA, arma::fill::zeros);
    testing_var = make_unique<vec_t>(NUM_TESTING_DATA, arma::fill::zeros);
}

void model::train(const mat_t &input, const vec_t &target)
{
    //mean_func(training_mean.get(), input);
    COV_FUNC->comp_ker(training_cov.get(), input, input);
    for (int i = 0; i < NUM_TRAINING_DATA; ++i)
        (*training_cov)(i, i) += VAR_NOISE;
    *chol_cov = arma::chol(*training_cov);
    *alpha = arma::solve(arma::trimatl(chol_cov->t()), target);
    *alpha = arma::solve(arma::trimatu(*chol_cov), *alpha);
}

pair<const vec_t *, const vec_t *> model::regress(const mat_t &training_input, const mat_t &testing_input)
{
    for (int i = 0; i < testing_input.n_cols; ++i)
    {
        COV_FUNC->comp_ker(mixture_cov.get(), training_input, testing_input.col(i));
        COV_FUNC->comp_ker(testing_cov.get(), testing_input.col(i), testing_input.col(i));
        (*testing_cov)(0, 0) += VAR_NOISE;
        *v = arma::solve(arma::trimatl(chol_cov->t()), *mixture_cov);
    
        (*testing_target)(i) = arma::sum(mixture_cov->t() * *alpha);
        (*testing_var)(i) = arma::sum(*testing_cov - v->t() * *v);
    }

    return make_pair(testing_target.get(), testing_var.get());
}

} // namespace gp

#ifdef UNIT_TESTING

int main()
{
    const string training_data_path{"/Users/evensong/ml_data/regression/sinc_sample"};
    const string testing_data_path{"/Users/evensong/ml_data/regression/sinc_testing_sample"};
    auto training_data(artificial_data::read_data(training_data_path));
    auto testing_data(artificial_data::read_data(testing_data_path));

    gp::init_arg gp_args;
    gp_args.num_training_data = training_data->second.size();
    gp_args.num_testing_data = testing_data->second.size();
    gp_args.var_noise = 0.01;
    gp_args.cov_func = make_unique<kernel_func::mixture_ker>(1.0, 1.0, 0.0, 0.0);

    gp::model gp_reg(&gp_args);
    gp_reg.train(training_data->first, training_data->second);
    auto result = gp_reg.regress(training_data->first, testing_data->first);
    cout << "target: " << testing_data->second.t() << endl;
    cout << "result: " << result.first->t() << endl;
    cout << "variance: " << result.second->t() << endl;

    return 0;
}

#endif

