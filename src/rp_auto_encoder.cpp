#include "../include/auto_encoder.h"

//#define VALIDATE
#define UNIT_TESTING
//#define DEBUG

namespace auto_encoder
{

namespace rp_auto_encoder
{

batch_network::batch_network(rp_auto_encoder::batch_init_arg *args) :
    auto_encoder::batch_network(&args->ae_args)
{
    arma::arma_rng::set_seed_random();
    check_arg(*args);
    set_arg(*args);

    clus = make_unique<vector<clustering::rpccl::cluster>>();
    clus->reserve(2);
    clustering::rpccl::init_arg clustering_args;
    for (int i = 0; i < 2; ++i)
    {
        clustering_args.max_times = 15;
        clustering_args.num_training_data = NUM_TRAINING_DATA;
        clustering_args.num_init_cluster = NUM_INIT_CLUSTER;
        clustering_args.numeric_dim = 1;
        clustering_args.penalty_rate = 1e-4;
        clustering_args.need_residue = true;
        clustering_args.input_row = weight[i]->n_rows;
        clustering_args.input_col = weight[i]->n_cols;
        clus->emplace_back(clustering_args);
        (*clus)[i].train(*weight[i]);
    }
    /*
    clus = make_unique<vector<clustering::cpcl::cluster>>();
    clus->reserve(2);

    clustering::cpcl::init_arg clustering_args;
    for (int i = 0; i < 2; ++i)
    {
    clustering_args.max_epoch = 10;
    clustering_args.num_cluster = NUM_INIT_CLUSTER;
    clustering_args.penalty_rate = 1e-3;
        clustering_args.input_row = weight[i]->n_rows;
        clustering_args.input_col = weight[i]->n_cols;
        clus->emplace_back(clustering_args);
        (*clus)[i].train(*weight[i]);
    }
    */
}

double batch_network::data_func(const mat_t &input, mf_t *new_grad, const bool need_obj)
{
    const mat_t *output_residue = auto_encoder::batch_network::comp_output_residue(input);
    double obj = 0.0;
    
    if (need_obj)
    {
        obj = std::accumulate(output_residue->begin(), output_residue->end(), 0.0, [](const double lhs, const double rhs){ return lhs + std::pow(rhs, 2); });
        obj *= 0.5 / NUM_TRAINING_DATA;
        obj += comp_obj_weight_decay();
    }

    if (new_grad != nullptr)
    {
        const mat_t *layer1_residue = single_bp((*output)[0], *output_residue, 1, new_grad);
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
    plus_grad_weight_decay(w_grad, dst_layer);
    *b_grad = arma::sum((*residue)[dst_layer], 1) / NUM_TRAINING_DATA;
}

double batch_network::comp_obj_weight_decay()
{
    double regu_term = 0.0;
    for (int i = 0; i < 2; ++i)
    {
        double tmp = (*clus)[i].get_obj_residue();
        if (!(*clus)[0].is_valid_data(tmp))
            CRY("invalid obj residue");
        regu_term += tmp;
    }
    regu_term *= WEIGHT_DECAY_RATE;

    return regu_term;
}

void batch_network::plus_grad_weight_decay(mat_t *w_grad, const int dst_layer)
{
    const mat_t *grad_residue = (*clus)[dst_layer].get_grad_residue();
    *w_grad += WEIGHT_DECAY_RATE * *grad_residue;
}

void batch_network::check_grad()
{
    cout << "rpccl check\n";
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
    (*num_grad)(1, 2).ones();

    const double eps = 1e-4;

    mf_for_each(num_grad.get(), field_args.get(), [eps, &tmp_input, this](mat_t &dst, mat_t &src){
            m_for_each(&dst, &src, [eps, &tmp_input, this](double &val_dst, double &val_src){
                    val_src += eps;
                    after_update_func();
                    val_dst = data_func(tmp_input, nullptr);
                    val_src -= 2 * eps;
                    after_update_func();
                    double tmp = data_func(tmp_input, nullptr);
                    val_dst -= tmp;
                    val_dst /= eps * 2.0;
                    val_src += eps;
                    after_update_func();
            });
    });
    data_func(tmp_input, analy_grad.get(), false);
    unique_ptr<mf_t> diff_tmp = make_unique<mf_t>(2, 1);
    for (int r = 0; r < 2; ++r)
    {
        const int c = 0;
        (*diff_tmp)(r, 0) = (*num_grad)(r, c) - (*analy_grad)(r, c);
    }
    const double diff1 = mf_norm2(*diff_tmp);
    cout << "diff1: " << diff1 << endl;
}

void mnist_rpccl_ae_test()
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

    unique_ptr<auto_encoder::stacked_ae_init_arg<rp_auto_encoder::batch_init_arg, softmax::batch::init_arg>> sae_args = 
        make_unique<auto_encoder::stacked_ae_init_arg<rp_auto_encoder::batch_init_arg, softmax::batch::init_arg>>();
    sae_args->num_layer = 2;
    sae_args->max_epoch = 2000;
    sae_args->num_training_data = mnist::NUM_TRAINING_IMAGE;
#ifdef VALIDATE
    sae_args->num_testing_data = mnist::NUM_VALIDATION_IMAGE;
#else
    sae_args->num_testing_data = mnist::NUM_TESTING_IMAGE;
#endif
    int dim[3] = {mnist::IMAGE_ROWSIZE * mnist::IMAGE_COLSIZE, 200, 200};
    int num_cluster[2] = {30, 10};

    for (int i = 0; i < sae_args->num_layer; ++i)
    {
        sae_args->sub_ae_args.emplace_back();
        sae_args->sub_ae_args.back().num_init_cluster = num_cluster[i];
        sae_args->sub_ae_args.back().ae_args.max_epoch = 200;
        sae_args->sub_ae_args.back().ae_args.num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().ae_args.num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().ae_args.dim_input = dim[i];
        sae_args->sub_ae_args.back().ae_args.dim_encode_layer = dim[i + 1];
        sae_args->sub_ae_args.back().ae_args.weight_decay_rate = 1e-4; 
        sae_args->sub_ae_args.back().ae_args.tuning_weight_decay_rate = 0.0; 
        sae_args->sub_ae_args.back().ae_args.act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.weight_decay_rate = 1e-4;
    sae_args->clsf_arg.max_epoch = 200;
    sae_args->clsf_arg.dim_input = sae_args->sub_ae_args.back().ae_args.dim_encode_layer;
    sae_args->clsf_arg.dim_output = mnist::NUM_LABEL_TYPE;
    sae_args->clsf_arg.num_training_data = sae_args->num_training_data;
    sae_args->clsf_arg.num_testing_data = sae_args->num_testing_data;

    unique_ptr<auto_encoder::stacked_ae<rp_auto_encoder::batch_network, softmax::batch::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<rp_auto_encoder::batch_network, softmax::batch::classifier>>(sae_args.get());
#ifdef VALIDATE
    sae->train(*training_image, *training_label, validation_image.get(), validation_label.get());
#else
    sae->train(*training_image, *training_label, testing_image.get(), testing_label.get());
#endif
}

void uci_rpccl_ae_test()
{
    const string ATTR_PATH{"/Users/evensong/ml_data/uci/adult/attr_name"};
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/uci/adult/training_data"};
#ifdef VALIDATE
    const string VALIDATION_DATA_PATH{"/Users/evensong/ml_data/uci/adult/validation_data"};
#endif
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/uci/adult/testing_data"};

    unique_ptr<uci::attr_info> attr(uci::read_attr(ATTR_PATH));
    unique_ptr<uci::classification_data> training_data(uci::read_data(TRAINING_DATA_PATH, *attr));
#ifdef VALIDATE
    unique_ptr<uci::classification_data> validation_data(uci::read_data(VALIDATION_DATA_PATH, *attr));
#endif
    unique_ptr<uci::classification_data> testing_data(uci::read_data(TESTING_DATA_PATH, *attr));
    uci::normalize(training_data.get());
#ifdef VALIDATE
    uci::normalize(validation_data.get());
#endif
    uci::normalize(testing_data.get());

    unique_ptr<auto_encoder::stacked_ae_init_arg<rp_auto_encoder::batch_init_arg, softmax::batch::init_arg>> sae_args = 
        make_unique<auto_encoder::stacked_ae_init_arg<rp_auto_encoder::batch_init_arg, softmax::batch::init_arg>>();
    sae_args->num_layer = 1;
    sae_args->max_epoch = 400;
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
        sae_args->sub_ae_args.back().num_init_cluster = 50;
        sae_args->sub_ae_args.back().ae_args.max_epoch = 200;
        sae_args->sub_ae_args.back().ae_args.num_training_data = sae_args->num_training_data;
        sae_args->sub_ae_args.back().ae_args.num_testing_data = sae_args->num_testing_data;
        sae_args->sub_ae_args.back().ae_args.dim_input = dim[i];
        sae_args->sub_ae_args.back().ae_args.dim_encode_layer = dim[i + 1];
        //sae_args->sub_ae_args.back().ae_args.weight_decay_rate = 1e-4; 
        sae_args->sub_ae_args.back().ae_args.weight_decay_rate = 1e-4; 
        sae_args->sub_ae_args.back().ae_args.tuning_weight_decay_rate = 0.0; 
        sae_args->sub_ae_args.back().ae_args.act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();
    }
    sae_args->clsf_arg.weight_decay_rate = 1e-4;
    sae_args->clsf_arg.max_epoch = 200;
    sae_args->clsf_arg.dim_input = sae_args->sub_ae_args.back().ae_args.dim_encode_layer;
    sae_args->clsf_arg.dim_output = attr->dim_output;
    sae_args->clsf_arg.num_training_data = sae_args->num_training_data;
    sae_args->clsf_arg.num_testing_data = sae_args->num_testing_data;

    unique_ptr<auto_encoder::stacked_ae<rp_auto_encoder::batch_network, softmax::batch::classifier>> sae =
       make_unique<auto_encoder::stacked_ae<rp_auto_encoder::batch_network, softmax::batch::classifier>>(sae_args.get());
#ifdef VALIDATE
    sae->train(training_data->input, training_data->output, &validation_data->input, &validation_data->output);
#else
    sae->train(training_data->input, training_data->output, &testing_data->input, &testing_data->output);
#endif
}


} // namespace rp_auto_encoder

} // namespace auto_encoder

#ifdef UNIT_TESTING
int main()
{
    cout << "rp_ae\n";
    auto_encoder::rp_auto_encoder::uci_rpccl_ae_test();
    return 0;
}
#endif


