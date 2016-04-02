#include "../include/auto_encoder.h"

//#define VALIDATE
#define UNIT_TESTING 
//#define DEBUG
//#define SPARSITY 

namespace auto_encoder
{

batch_network::batch_network(batch_init_arg *args)
{
    arma::arma_rng::set_seed_random();
    check_arg(*args);
    set_arg(args);

    output = make_unique<vector<mat_t>>();
    drv = make_unique<vector<mat_t>>();
    residue = make_unique<vector<mat_t>>();
    testing_output = make_unique<mat_t>(DIM_INPUT, NUM_TESTING_DATA, arma::fill::zeros);
    for (int i = 0; i < 2; ++i)
    {
        weight.push_back(nullptr);
        bias.push_back(nullptr);
    }

    field_args = make_unique<mf_t>(2, 2);
    init_field_arg();
    set_field_ptr(*field_args);

    optimization::l_bfgs::init_arg optm_args;
    optm_args.max_epoch = MAX_EPOCH;
    optm = make_unique<optimization::wrapper<mf_t, optimization::l_bfgs::optimizer<mf_t>>>(optm_args, *field_args);

    output->push_back(mat_t(DIM_ENCODE_LAYER, NUM_TRAINING_DATA, arma::fill::zeros));
    output->push_back(mat_t(DIM_INPUT, NUM_TRAINING_DATA, arma::fill::zeros));
    drv->push_back(mat_t(DIM_ENCODE_LAYER, NUM_TRAINING_DATA, arma::fill::zeros));
    drv->push_back(mat_t(DIM_INPUT, NUM_TRAINING_DATA, arma::fill::zeros));
    residue->push_back(mat_t(DIM_ENCODE_LAYER, NUM_TRAINING_DATA, arma::fill::zeros));
    residue->push_back(mat_t(DIM_INPUT, NUM_TRAINING_DATA, arma::fill::zeros));
    prv_residue = make_unique<mat_t>(DIM_INPUT, NUM_TRAINING_DATA, arma::fill::zeros);
}

void batch_network::check_arg(const batch_init_arg &args)
{
    if (args.max_epoch <= 0)
        CRY("invalid max_epoch");
    if (args.num_training_data <= 0)
        CRY("invalid num_training_data");
    if (args.num_testing_data <= 0)
        CRY("invalid num_testing_data");
    if (args.dim_input <= 0)
        CRY("invalid dim_input");
    if (args.dim_encode_layer <= 0)
        CRY("invalid dim_encode_layer");
    if (args.act_function == nullptr)
        CRY("empty act_function");
}

void batch_network::set_arg(batch_init_arg *args)
{
    DIM_INPUT = args->dim_input;
    NUM_TRAINING_DATA = args->num_training_data;
    NUM_TESTING_DATA = args->num_testing_data;
    MAX_EPOCH = args->max_epoch;
    WEIGHT_DECAY_RATE = args->weight_decay_rate;
    TUNING_WEIGHT_DECAY_RATE = args->tuning_weight_decay_rate;
    DIM_ENCODE_LAYER = args->dim_encode_layer;
    //SPARSITY_DECAY_RATE = args->sparsity_decay_rate;
    //SPARSITY_PARAMETER = args->sparsity_parameter;
    ACT_FUNCTION = std::move(args->act_function);
}

const mat_t *batch_network::single_fp(const mat_t &input, const int num_input, const int dst_layer)
{
    mat_t *result = nullptr;
    if (num_input == NUM_TRAINING_DATA)
        result = &(*output)[dst_layer];
    else if (num_input == NUM_TESTING_DATA)
        result = testing_output.get();
    else
        CRY("invalid num_input");

        *result = *weight[dst_layer] * input;
        result->each_col() += *bias[dst_layer];
        result->transform([this](double e){ return ACT_FUNCTION->obj(e); });
        if (num_input == NUM_TRAINING_DATA)
            std::transform(result->begin(), result->end(), (*drv)[dst_layer].begin(),
                    [this](const double e){ return ACT_FUNCTION->drv_wrt_result(e); });

    return result;
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

double batch_network::data_func(const mat_t &input, mf_t *new_grad, const bool need_obj)
{
    const mat_t *output_residue = comp_output_residue(input);
    double obj = 0.0;
    
    if (need_obj)
    {
        obj = std::accumulate(output_residue->begin(), output_residue->end(), 0.0, [](const double lhs, const double rhs){ return lhs + std::pow(rhs, 2); });
        obj *= 0.5 / NUM_TRAINING_DATA;
        if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
            obj += WEIGHT_DECAY_RATE * 0.5 * (arma::accu(arma::pow(*weight[0], 2)) + arma::accu(arma::pow(*weight[1], 2)));
    }

    if (new_grad != nullptr)
    {
        const mat_t *layer1_residue = single_bp((*output)[0], *output_residue, 1, new_grad);
        single_bp(input, *layer1_residue, 0, new_grad);
    }

    return obj;
}

void batch_network::comp_grad(const mat_t &forward_result, const int dst_layer, mf_t *new_grad)
{
    mat_t *new_w_grad = &(*new_grad)(0, dst_layer);
    mat_t *new_b_grad = &(*new_grad)(1, dst_layer);
    *new_w_grad = (*residue)[dst_layer] * forward_result.t() / NUM_TRAINING_DATA;
    if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
        *new_w_grad += WEIGHT_DECAY_RATE * *weight[dst_layer];
    *new_b_grad = arma::sum((*residue)[dst_layer], 1) / NUM_TRAINING_DATA;
}

void batch_network::check_grad(const mat_t &input)
{
    cout << "ae check\n";
    using namespace field_algorithm;

    mat_t tmp_input(60, 3, arma::fill::randu);
    unique_ptr<mf_t> num_grad = make_unique<mf_t>(2, 2);
    unique_ptr<mf_t> analy_grad = make_unique<mf_t>(2, 2);
    unique_ptr<mf_t> diff1_grad = make_unique<mf_t>(2, 2);
    unique_ptr<mf_t> diff2_grad = make_unique<mf_t>(2, 2);
    mf_init(num_grad.get(), *field_args);
    mf_init(analy_grad.get(), *field_args);
    mf_init(diff1_grad.get(), *field_args);
    mf_init(diff2_grad.get(), *field_args);

    const double eps = 1e-4;
    int i = 1;

    mf_for_each(num_grad.get(), field_args.get(), [&i, eps, &tmp_input, this](mat_t &dst, mat_t &src){
            m_for_each(&dst, &src, [&i, eps, &tmp_input, this](double &val_dst, double &val_src){
                    ++i;
                    if (i % 20000 == 0)
                        cout << "epoch: " << i << endl;
                    val_src += eps;
                    val_dst = data_func(tmp_input, nullptr);
                    val_src -= 2 * eps;
                    val_dst -= data_func(tmp_input, nullptr);
                    val_dst /= eps * 2.0;
                    val_src += eps;
            });
    });
    data_func(tmp_input, analy_grad.get(), false);

    unique_ptr<mf_t> diff_tmp = make_unique<mf_t>(2, 1);
    for (int r = 0; r < 2; ++r)
    {
        const int c = 1;
        (*diff_tmp)(r, 0) = (*num_grad)(r, c) - (*analy_grad)(r, c);
    }
    const double diff1 = mf_norm2(*diff_tmp);
    cout << "diff1: " << diff1 << endl;
    //mf_for_each(diff1_grad.get(), *num_grad, *analy_grad, [](mat_t &dst, const mat_t &src1, const mat_t &src2){ dst = src1 - src2; });
    //const double diff1 = mf_norm2(*diff1_grad);
    //mf_for_each(diff2_grad.get(), *num_grad, *analy_grad, [](mat_t &dst, const mat_t &src1, const mat_t &src2){ dst = src1 + src2; });
    //const double diff2 = mf_norm2(*diff2_grad);
    //cout << "\n\ndiff: " << diff1 / diff2 << '\t' << diff1 << '\t' << diff2 << endl;
}

void mnist_auto_encoder_test()
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

    //ae_args.sparsity_decay_rate = 3;
    //ae_args.sparsity_parameter = 0.01;
    unique_ptr<auto_encoder::stacked_ae_init_arg<auto_encoder::batch_init_arg, softmax::batch::init_arg>> sae_args = 
        make_unique<auto_encoder::stacked_ae_init_arg<auto_encoder::batch_init_arg, softmax::batch::init_arg>>();
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
        sae_args->sub_ae_args.back().max_epoch = 50;
        sae_args->sub_ae_args.back().num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().dim_input = dim[i];
        sae_args->sub_ae_args.back().dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().weight_decay_rate = 3e-3; // 0.001: after about 10 epochs, accuracy is still worse about 0.5%
                                                               // 0.003: best
                                                               // 0.005: worse than 0.001
        sae_args->sub_ae_args.back().tuning_weight_decay_rate = 1e-4; 
        sae_args->sub_ae_args.back().act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.weight_decay_rate = 1e-4;
    sae_args->clsf_arg.max_epoch = 50;
    sae_args->clsf_arg.dim_input = sae_args->sub_ae_args.back().dim_encode_layer;
    sae_args->clsf_arg.dim_output = mnist::NUM_LABEL_TYPE;
    sae_args->clsf_arg.num_training_data = sae_args->num_training_data;
    sae_args->clsf_arg.num_testing_data = sae_args->num_testing_data;

    unique_ptr<auto_encoder::stacked_ae<auto_encoder::batch_network, softmax::batch::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<auto_encoder::batch_network, softmax::batch::classifier>>(sae_args.get());
    sae->train(*training_image, *training_label, testing_image.get(), testing_label.get());
    //sae->train(*training_image, *training_label, validation_image.get(), validation_label.get());

}

void uci_auto_encoder_test()
{
    const string ATTR_PATH{"/Users/evensong/ml_data/uci/adult/attr_name"};
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/uci/adult/training_data"};
    const string VALIDATION_DATA_PATH{"/Users/evensong/ml_data/uci/adult/validation_data"};
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/uci/adult/testing_data"};

    unique_ptr<uci::attr_info> attr(uci::read_attr(ATTR_PATH));
    unique_ptr<uci::classification_data> training_data(uci::read_data(TRAINING_DATA_PATH, *attr));
    unique_ptr<uci::classification_data> validation_data(uci::read_data(VALIDATION_DATA_PATH, *attr));
    unique_ptr<uci::classification_data> testing_data(uci::read_data(TESTING_DATA_PATH, *attr));
    uci::normalize(training_data.get());
    uci::normalize(validation_data.get());
    uci::normalize(testing_data.get());

    unique_ptr<auto_encoder::stacked_ae_init_arg<auto_encoder::batch_init_arg, softmax::batch::init_arg>> sae_args = 
    make_unique<auto_encoder::stacked_ae_init_arg<auto_encoder::batch_init_arg, softmax::batch::init_arg>>();
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
        sae_args->sub_ae_args.back().max_epoch = 200;
        sae_args->sub_ae_args.back().num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().dim_input = dim[i];
        sae_args->sub_ae_args.back().dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().weight_decay_rate = 2e-4; 
        sae_args->sub_ae_args.back().tuning_weight_decay_rate = 0.0; 
        sae_args->sub_ae_args.back().act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.weight_decay_rate = 1e-4;
    sae_args->clsf_arg.max_epoch = 200;
    sae_args->clsf_arg.dim_input = sae_args->sub_ae_args.back().dim_encode_layer;
    sae_args->clsf_arg.dim_output = attr->dim_output;
    sae_args->clsf_arg.num_training_data = sae_args->num_training_data;
    sae_args->clsf_arg.num_testing_data = sae_args->num_testing_data;

    unique_ptr<auto_encoder::stacked_ae<auto_encoder::batch_network, softmax::batch::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<auto_encoder::batch_network, softmax::batch::classifier>>(sae_args.get());
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

    unique_ptr<auto_encoder::stacked_ae_init_arg<auto_encoder::batch_init_arg, knn::init_arg>> sae_args = 
    make_unique<auto_encoder::stacked_ae_init_arg<auto_encoder::batch_init_arg, knn::init_arg>>();

    sae_args->num_layer = 1;
    sae_args->max_epoch = 300;
    sae_args->num_testing_data = training_data->input.n_cols;
    sae_args->num_training_data = training_data->input.n_cols;
    sae_args->is_simple_training = true;
    int dim[2] = {attr->dim_input, 2};

    for (int i = 0; i < sae_args->num_layer; ++i)
    {
        sae_args->sub_ae_args.emplace_back();
        sae_args->sub_ae_args.back().max_epoch = 200;
        sae_args->sub_ae_args.back().num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().dim_input = dim[i];
        sae_args->sub_ae_args.back().dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().weight_decay_rate = 1e-4; 
        sae_args->sub_ae_args.back().tuning_weight_decay_rate = 0.0; 
        sae_args->sub_ae_args.back().act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.num_neighbor = 5;
    sae_args->clsf_arg.dim_output = attr->dim_output;
    sae_args->clsf_arg.num_training_data = training_data->input.n_cols;
    sae_args->clsf_arg.num_testing_data = training_data->input.n_cols;
    unique_ptr<auto_encoder::stacked_ae<auto_encoder::batch_network, knn::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<auto_encoder::batch_network, knn::classifier>>(sae_args.get());
    sae->simple_train(training_data->input);
    sae->testing(training_data->input, training_data->output, true);
}

}

/*
 * mnist learning result:
 * ae: weight decay = 3e-3, DIM_ENCODE_LAYER = {200, 200}
 * softmax: weight decay = 1e-4
 *
 * l-bfgs:
 * pre-training: autoencoder 200 epochs, softmax 200 epochs
 * tune:
 * epoch = 1: 92.95%
 * epoch = 10: 93.9%
 * epoch = 20: 94.95%
 * epoch = 30: 95.86%
 * epoch = 40: 96.21%
 * epoch = 50: 96.72%
 * epoch = 60: 96.87%
 * epoch = 70: 97.12%
 * epoch = 80: 97.34%
 * epoch = 90: 97.46%
 * epoch = 100: 97.62%
 * epoch = 110: 97.64%
 * epoch = 116: 97.81%, tuning finished, 19min 31sec in total
 */
#ifdef UNIT_TESTING
int main()
{
    auto_encoder::uci_auto_encoder_test();
    //auto_encoder::uci_data_compress_test();
    return 0;
}
#endif
