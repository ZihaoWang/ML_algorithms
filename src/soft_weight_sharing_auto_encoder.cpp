#include "../include/auto_encoder.h"

//#define VALIDATE
#define UNIT_TESTING
//#define DEBUG

namespace auto_encoder
{

namespace sws_auto_encoder
{

batch_network::batch_network(sws_auto_encoder::batch_init_arg *args) :
    auto_encoder::batch_network(&args->ae_args)
{
    arma::arma_rng::set_seed_random();
    check_arg(*args);
    set_arg(*args);

    for (int i = 0; i < 2; ++i)
    {
        mean.push_back(nullptr);
        aux_variance.push_back(nullptr);
        aux_prior.push_back(nullptr);
    }
    variance = make_unique<vector<vec_t>>(2, vec_t(NUM_GAUSSIAN));
    prior = make_unique<vector<vec_t>>(2, vec_t(NUM_GAUSSIAN));
    posterior = make_unique<vector<cube_t>>();

    gm_field_args = make_unique<mf_t>(3, 2);
    gm_init_field_arg();
    gm_set_field_ptr(*gm_field_args);

    optimization::l_bfgs::init_arg gm_optm_args;
    gm_optm_args.max_epoch = 1;
    gm_optm = make_unique<optimization::wrapper<mf_t, optimization::l_bfgs::optimizer<mf_t>>>(gm_optm_args, *gm_field_args);

    posterior->reserve(2);
    posterior->emplace_back(cube_t(DIM_ENCODE_LAYER, DIM_INPUT, NUM_GAUSSIAN, arma::fill::zeros));
    posterior->emplace_back(cube_t(DIM_INPUT, DIM_ENCODE_LAYER, NUM_GAUSSIAN, arma::fill::zeros));
    for (int i = 0; i < 2; ++i)
    {
        comp_variance(i);
        comp_prior(i);
        comp_posterior(i);
    }
}

void batch_network::gm_init_field_arg()
{
    (*gm_field_args)(0, 0) = vec_t(NUM_GAUSSIAN, arma::fill::randu);
    (*gm_field_args)(1, 0) = vec_t(NUM_GAUSSIAN, arma::fill::randu);
    (*gm_field_args)(2, 0) = vec_t(NUM_GAUSSIAN, arma::fill::randu);
    (*gm_field_args)(0, 1) = vec_t(NUM_GAUSSIAN, arma::fill::randu);
    (*gm_field_args)(1, 1) = vec_t(NUM_GAUSSIAN, arma::fill::randu);
    (*gm_field_args)(2, 1) = vec_t(NUM_GAUSSIAN, arma::fill::randu);
}

void batch_network::gm_set_field_ptr(const mf_t &gm_f)
{
    mean[0] = &gm_f(0, 0);
    aux_variance[0] = &gm_f(1, 0);
    aux_prior[0] = &gm_f(2, 0);
    mean[1] = &gm_f(0, 1);
    aux_variance[1] = &gm_f(1, 1);
    aux_prior[1] = &gm_f(2, 1);
}

pair<double, const mat_t *> batch_network::comp_obj(const mat_t &input)
{
    const mat_t *output_residue = auto_encoder::batch_network::comp_output_residue(input);
    
    double obj = std::accumulate(output_residue->begin(), output_residue->end(), 0.0, [](const double lhs, const double rhs){ return lhs + std::pow(rhs, 2); });
    obj *= 0.5 / NUM_TRAINING_DATA;
    if (std::abs(WEIGHT_DECAY_RATE) > DBL_EPSILON)
        obj += minus_obj_weight_decay();

    return make_pair(obj, output_residue);
}

double batch_network::data_func(const mat_t &input, mf_t *new_grad, const bool need_obj)
{
    double obj = 0.0;
    auto result = comp_obj(input);
    if (need_obj)
        obj = result.first;

    if (new_grad != nullptr)
    {
        const mat_t *layer1_residue = single_bp((*output)[0], *result.second, 1, new_grad);
        single_bp(input, *layer1_residue, 0, new_grad);
    }

    return obj;
}

const mat_t *batch_network::single_bp(const mat_t &input, const mat_t &backward_residue, const int dst_layer, mf_t *new_grad)
{
    const mat_t *result = nullptr;
    (*residue)[dst_layer] = (*drv)[dst_layer] % backward_residue;
    if (dst_layer == 1)
    {
        (*residue)[0] = weight[1]->t() * (*residue)[1];
        result = &(*residue)[0];
    }
    else
    {
        *prv_residue = weight[0]->t() * (*residue)[0];
        result = prv_residue.get();
    }
    comp_grad(input, dst_layer, new_grad);

    return result;
}

void batch_network::comp_grad(const mat_t &forward_result, const int dst_layer, mf_t *new_grad)
{
    mat_t *w_grad = &(*new_grad)(0, dst_layer);
    mat_t *b_grad = &(*new_grad)(1, dst_layer);
    *w_grad = (*residue)[dst_layer] * forward_result.t() / NUM_TRAINING_DATA;
    if (std::abs(WEIGHT_DECAY_RATE) > DBL_EPSILON)
        plus_grad_weight_decay(w_grad, dst_layer);
    *b_grad = arma::sum((*residue)[dst_layer], 1) / NUM_TRAINING_DATA;
}

double batch_network::minus_obj_weight_decay()
{
    double regu_term = 0.0;   
    for (int i = 0; i < 2; ++i)
        regu_term += std::accumulate(weight[i]->begin(), weight[i]->end(), 0.0, [this, i](const double lhs, const double rhs)
        {
            double tmp = 0.0;
            for (int j = 0; j < NUM_GAUSSIAN; ++j)
                tmp += (*prior)[i](j) * comp_gaussian(rhs, i, j);
            return lhs + std::log(tmp);
        });
    regu_term *= WEIGHT_DECAY_RATE;
    regu_term *= -1;

    return regu_term;
}

void batch_network::plus_grad_weight_decay(mat_t *w_grad, const int dst_layer)
{
    for (int i = 0; i < weight[dst_layer]->n_rows; ++i)
        for (int j = 0; j < weight[dst_layer]->n_cols; ++j)
        {
            double w_decay = 0.0;
            for (int k = 0; k < NUM_GAUSSIAN; ++k)
                w_decay += (*posterior)[dst_layer](i, j, k) * ((*weight[dst_layer])(i, j) - (*mean[dst_layer])[k]) / (*variance)[dst_layer][k];
            (*w_grad)(i, j) += WEIGHT_DECAY_RATE * w_decay;
        }
    //*w_grad += WEIGHT_DECAY_RATE * *weight[dst_layer];
}

double batch_network::gm_data_func(const mat_t &input, mf_t *new_grad, const bool need_obj)
{
    double obj = 0.0;
    if (need_obj)
        obj = comp_obj(input).first;

    if (new_grad != nullptr)
    {
        gm_comp_grad(1, new_grad);
        gm_comp_grad(0, new_grad);
    }

    return obj;
}

void batch_network::gm_comp_grad(const int dst_layer, mf_t *new_grad)
{
    mat_t *m_grad = &(*new_grad)(0, dst_layer);
    mat_t *aux_v_grad = &(*new_grad)(1, dst_layer);
    mat_t *aux_p_grad = &(*new_grad)(2, dst_layer);
    m_grad->zeros();
    aux_v_grad->zeros();
    aux_p_grad->zeros();

    comp_posterior(dst_layer);
    for (int i = 0; i < weight[dst_layer]->n_rows; ++i)
        for (int j = 0; j < weight[dst_layer]->n_cols; ++j)
        {
            const double weight_now = (*weight[dst_layer])(i, j);
            for (int k = 0; k < NUM_GAUSSIAN; ++k)
            {
                const double posterior_now = (*posterior)[dst_layer](i, j, k);
                const double mean_now = (*mean[dst_layer])[k];
                const double variance_now = (*variance)[dst_layer][k];

                (*m_grad)[k] += posterior_now * (mean_now - weight_now) / variance_now; 
                (*aux_v_grad)[k] += 0.5 * posterior_now * (1.0 - std::pow((weight_now - mean_now), 2) / variance_now);
                (*aux_p_grad)[k] += (*prior)[dst_layer][k] - posterior_now;
            }
        }

    if (std::abs(WEIGHT_DECAY_RATE) > DBL_EPSILON)
    {
        *m_grad *= WEIGHT_DECAY_RATE;
        *aux_v_grad *= WEIGHT_DECAY_RATE;
    }

    comp_variance(dst_layer);
    comp_prior(dst_layer);
}

void batch_network::comp_posterior(const int dst_layer)
{
    for (int i = 0; i < weight[dst_layer]->n_rows; ++i)
        for (int j = 0; j < weight[dst_layer]->n_cols; ++j)
        {
            for (int k = 0; k < NUM_GAUSSIAN; ++k)
                (*posterior)[dst_layer](i, j, k) = (*prior)[dst_layer][k] * comp_gaussian((*weight[dst_layer])(i, j), dst_layer, k);
            (*posterior)[dst_layer].tube(i, j) /= arma::accu((*posterior)[dst_layer].tube(i, j));
        }
}

void mnist_gaussian_ae_test()
{
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/mnist/train-images-idx3-ubyte"};
    const string TRAINING_LABEL_PATH{"/Users/evensong/ml_data/mnist/train-labels-idx1-ubyte"};
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/mnist/t10k-images-idx3-ubyte"};
    const string TESTING_LABEL_PATH{"/Users/evensong/ml_data/mnist/t10k-labels-idx1-ubyte"};

    unique_ptr<mat_t> training_image(mnist::read_vectorized_image(TRAINING_DATA_PATH, mnist::set_t::TRAINING));
    unique_ptr<mat_t> validation_image(mnist::read_vectorized_image(TRAINING_DATA_PATH, mnist::set_t::VALIDATION));
    unique_ptr<mat_t> testing_image(mnist::read_vectorized_image(TESTING_DATA_PATH, mnist::set_t::TESTING));
    unique_ptr<vector<int>> training_label(mnist::read_label(TRAINING_LABEL_PATH, mnist::set_t::TRAINING));
    unique_ptr<vector<int>> validation_label(mnist::read_label(TRAINING_LABEL_PATH, mnist::set_t::VALIDATION));
    unique_ptr<vector<int>> testing_label(mnist::read_label(TESTING_LABEL_PATH, mnist::set_t::TESTING));
    mnist::normalize(training_image.get());
    mnist::normalize(validation_image.get());
    mnist::normalize(testing_image.get());

    unique_ptr<auto_encoder::stacked_ae_init_arg<sws_auto_encoder::batch_init_arg, softmax::batch::init_arg>> sae_args = 
        make_unique<auto_encoder::stacked_ae_init_arg<sws_auto_encoder::batch_init_arg, softmax::batch::init_arg>>();
    sae_args->num_layer = 2;
    sae_args->max_epoch = 1000;
#ifdef VALIDATE
    sae_args->num_testing_data = mnist::NUM_VALIDATION_IMAGE;
#else
    sae_args->num_testing_data = mnist::NUM_TESTING_IMAGE;
#endif
    sae_args->num_training_data = mnist::NUM_TRAINING_IMAGE;
    int dim[3] = {mnist::IMAGE_ROWSIZE * mnist::IMAGE_COLSIZE, 200, 200};

    for (int i = 0; i < sae_args->num_layer; ++i)
    {
        sae_args->sub_ae_args.emplace_back();
        sae_args->sub_ae_args.back().num_gaussian = 16;
        sae_args->sub_ae_args.back().ae_args.max_epoch = 50;
        sae_args->sub_ae_args.back().ae_args.num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().ae_args.num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().ae_args.dim_input = dim[i];
        sae_args->sub_ae_args.back().ae_args.dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().ae_args.weight_decay_rate = 3e-3; // 0.001: after about 10 epochs, accuracy is still worse about 0.5%
                                                               // 0.003: best
                                                               // 0.005: worse than 0.001
        sae_args->sub_ae_args.back().ae_args.tuning_weight_decay_rate = 1e-4; 
        sae_args->sub_ae_args.back().ae_args.act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.weight_decay_rate = 1e-4;
    sae_args->clsf_arg.max_epoch = 50;
    sae_args->clsf_arg.dim_input = sae_args->sub_ae_args.back().ae_args.dim_encode_layer;
    sae_args->clsf_arg.dim_output = mnist::NUM_LABEL_TYPE;
    sae_args->clsf_arg.num_training_data = sae_args->num_training_data;
    sae_args->clsf_arg.num_testing_data = sae_args->num_testing_data;

    unique_ptr<auto_encoder::stacked_ae<sws_auto_encoder::batch_network, softmax::batch::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<sws_auto_encoder::batch_network, softmax::batch::classifier>>(sae_args.get());
#ifdef VALIDATE
    sae->train(*training_image, *training_label, validation_image.get(), validation_label.get());
#else
    sae->train(*training_image, *training_label, testing_image.get(), testing_label.get());
#endif
}

void uci_gaussian_ae_test()
{
    cout << "uci test" << endl;
    const string ATTR_PATH{"/Users/evensong/ml_data/uci/letter/attr_name"};
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/uci/letter/training_data"};
    const string VALIDATION_DATA_PATH{"/Users/evensong/ml_data/uci/letter/validation_data"};
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/uci/letter/testing_data"};

    unique_ptr<uci::attr_info> attr(uci::read_attr(ATTR_PATH));
    unique_ptr<uci::classification_data> training_data(uci::read_data(TRAINING_DATA_PATH, *attr));
    unique_ptr<uci::classification_data> validation_data(uci::read_data(VALIDATION_DATA_PATH, *attr));
    unique_ptr<uci::classification_data> testing_data(uci::read_data(TESTING_DATA_PATH, *attr));
    uci::normalize(training_data.get());
    uci::normalize(validation_data.get());
    uci::normalize(testing_data.get());

    unique_ptr<auto_encoder::stacked_ae_init_arg<sws_auto_encoder::batch_init_arg, softmax::batch::init_arg>> sae_args = 
        make_unique<auto_encoder::stacked_ae_init_arg<sws_auto_encoder::batch_init_arg, softmax::batch::init_arg>>();
    sae_args->num_layer = 1;
    sae_args->max_epoch = 300;
#ifdef VALIDATE
    sae_args->num_testing_data = validation_data->input.n_cols;
#else
    sae_args->num_testing_data = testing_data->input.n_cols;
#endif
    sae_args->num_training_data = training_data->input.n_cols;
    int dim[2] = {attr->dim_input, 100};

    for (int i = 0; i < sae_args->num_layer; ++i)
    {
        sae_args->sub_ae_args.emplace_back();
        sae_args->sub_ae_args.back().num_gaussian = 50;
        sae_args->sub_ae_args.back().ae_args.max_epoch = 200;
        sae_args->sub_ae_args.back().ae_args.num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().ae_args.num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().ae_args.dim_input = dim[i];
        sae_args->sub_ae_args.back().ae_args.dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().ae_args.weight_decay_rate = 2e-4;
        sae_args->sub_ae_args.back().ae_args.tuning_weight_decay_rate = 0.0; 
        sae_args->sub_ae_args.back().ae_args.act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.weight_decay_rate = 1e-4;
    sae_args->clsf_arg.max_epoch = 200;
    sae_args->clsf_arg.dim_input = sae_args->sub_ae_args.back().ae_args.dim_encode_layer;
    sae_args->clsf_arg.dim_output = attr->dim_output;
    sae_args->clsf_arg.num_training_data = sae_args->num_training_data;
    sae_args->clsf_arg.num_testing_data = sae_args->num_testing_data;

    unique_ptr<auto_encoder::stacked_ae<sws_auto_encoder::batch_network, softmax::batch::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<sws_auto_encoder::batch_network, softmax::batch::classifier>>(sae_args.get());
#ifdef VALIDATE
    sae->train(training_data->input, training_data->output, &validation_data->input, &validation_data->output);
#else
    sae->train(training_data->input, training_data->output, &testing_data->input, &testing_data->output);
#endif
}

void uci_data_compress_test()
{
    const string ATTR_PATH{"/Users/evensong/ml_data/uci/mushroom/attr_name"};
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/uci/mushroom/training_data"};

    unique_ptr<uci::attr_info> attr(uci::read_attr(ATTR_PATH));
    unique_ptr<uci::classification_data> training_data(uci::read_data(TRAINING_DATA_PATH, *attr));
    uci::normalize(training_data.get());

    unique_ptr<auto_encoder::stacked_ae_init_arg<sws_auto_encoder::batch_init_arg, knn::init_arg>> sae_args = 
    make_unique<auto_encoder::stacked_ae_init_arg<sws_auto_encoder::batch_init_arg, knn::init_arg>>();

    sae_args->num_layer = 1;
    sae_args->max_epoch = 300;
    sae_args->num_testing_data = training_data->input.n_cols;
    sae_args->num_training_data = training_data->input.n_cols;
    sae_args->is_simple_training = true;
    int dim[2] = {attr->dim_input, 2};

    for (int i = 0; i < sae_args->num_layer; ++i)
    {
        sae_args->sub_ae_args.emplace_back();
        sae_args->sub_ae_args.back().num_gaussian = 20;
        sae_args->sub_ae_args.back().ae_args.max_epoch = 200;
        sae_args->sub_ae_args.back().ae_args.num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().ae_args.num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().ae_args.dim_input = dim[i];
        sae_args->sub_ae_args.back().ae_args.dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().ae_args.weight_decay_rate = 2e-4;
        sae_args->sub_ae_args.back().ae_args.tuning_weight_decay_rate = 0.0; 
        sae_args->sub_ae_args.back().ae_args.act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.num_neighbor = 5;
    sae_args->clsf_arg.dim_output = attr->dim_output;
    sae_args->clsf_arg.num_training_data = training_data->input.n_cols;
    sae_args->clsf_arg.num_testing_data = training_data->input.n_cols;
    unique_ptr<auto_encoder::stacked_ae<sws_auto_encoder::batch_network, knn::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<sws_auto_encoder::batch_network, knn::classifier>>(sae_args.get());
    sae->simple_train(training_data->input);
    sae->testing(training_data->input, training_data->output, true);
}

} // namespace sws_auto_encoder

} // namespace auto_encoder

#ifdef UNIT_TESTING
int main()
{
    cout << "sws ae\n";
    auto_encoder::sws_auto_encoder::uci_data_compress_test();
    return 0;
}
#endif


