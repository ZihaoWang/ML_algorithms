#include "../include/classifier.h"

//#define UNIT_TESTING

namespace softmax
{

namespace batch
{

classifier::classifier(const init_arg &args)
{
    check_arg(args);
    set_arg(args);
    arma::arma_rng::set_seed_random();
    field_args = make_unique<mf_t>(2, 1);
    (*field_args)(0, 0) = 0.005 * mat_t(DIM_OUTPUT, DIM_INPUT, arma::fill::randu);
    (*field_args)(1, 0) = 0.005 * vec_t(DIM_OUTPUT, arma::fill::randu);
    set_field_ptr(*field_args);
    output = make_unique<mat_t>(DIM_OUTPUT, NUM_TRAINING_DATA, arma::fill::zeros);
    testing_output = make_unique<mat_t>(DIM_OUTPUT, NUM_TESTING_DATA, arma::fill::zeros);
    prv_residue = make_unique<mat_t>(DIM_INPUT, NUM_TRAINING_DATA, arma::fill::zeros);

    optimization::l_bfgs::init_arg optm_args;
    optm_args.max_epoch = MAX_EPOCH;
    optm = make_unique<optimization::wrapper<mf_t, optimization::l_bfgs::optimizer<mf_t>>>(optm_args, *field_args);
}

const mat_t *classifier::fp(const mat_t &input, const int num_input)
{
    mat_t *result = nullptr;
    if (num_input == NUM_TRAINING_DATA)
        result = output.get();
    else if (num_input == NUM_TESTING_DATA)
        result = testing_output.get();
    else
        CRY("invalid num_input");

    *result = *weight * input;
    result->each_col() += *bias;
    for (int i = 0; i < num_input; ++i)
        result->col(i) -= result->col(i).max();
    *result = arma::exp(*result);
    for (int i = 0; i < num_input; ++i)
        result->col(i) /= arma::accu(result->col(i));

    return result;
}
    
double classifier::data_func(const mat_t &input, const vector<int> &target, mf_t *new_grad, const bool need_obj)
{
    double obj = 0.0;
    predict(input);

    if (need_obj)
    {
        for (int i = 0; i < NUM_TRAINING_DATA; ++i)
            obj += std::log((*output)(target[i], i));
        obj /= -NUM_TRAINING_DATA;
        if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
            obj += WEIGHT_DECAY_RATE / 2 * arma::accu(arma::pow(*weight, 2));
    }

    if (new_grad != nullptr)
    {
        const mat_t *output_residue = comp_output_residue(target);
        mat_t *new_w_grad = &(*new_grad)(0, 0);
        mat_t *new_b_grad = &(*new_grad)(1, 0);
        *new_w_grad = *output_residue * input.t() / NUM_TRAINING_DATA;
        *new_b_grad = arma::sum(*output_residue, 1) / NUM_TRAINING_DATA;
    
        if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
            *new_w_grad += WEIGHT_DECAY_RATE * *weight;
    }

    return obj;
}

double classifier::testing(const mat_t &input, const vector<int> &target, const bool is_training_data, const bool need_print)
{
    int num_input;
    if (is_training_data)
        num_input = NUM_TRAINING_DATA;
    else
        num_input = NUM_TESTING_DATA;
    const mat_t *final_output = fp(input, num_input);
    int num_correct = 0;
    for (int idx_output = 0; idx_output < num_input; ++idx_output)
    {
        arma::uword idx_max;
        final_output->col(idx_output).max(idx_max);
        if (idx_max == target[idx_output])
            ++num_correct;
    }
    
    double accuracy = static_cast<double>(num_correct) / num_input;
    if (need_print)
    {
        if (is_training_data)
            cout << "training_data ";
        else
            cout << "testing_data ";
        cout << "testing accuracy: " << accuracy << endl;
    }

    return accuracy;
}

void batch_softmax_test()
{
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/mnist/train-images-idx3-ubyte"};
    const string TRAINING_LABEL_PATH{"/Users/evensong/ml_data/mnist/train-labels-idx1-ubyte"};
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/mnist/t10k-images-idx3-ubyte"};
    const string TESTING_LABEL_PATH{"/Users/evensong/ml_data/mnist/t10k-labels-idx1-ubyte"};

    unique_ptr<mat_t> training_image(mnist::read_vectorized_image(TRAINING_DATA_PATH, mnist::set_t::TRAINING));
    unique_ptr<mat_t> testing_image(mnist::read_vectorized_image(TESTING_DATA_PATH, mnist::set_t::TESTING));
    unique_ptr<vector<int>> training_label(mnist::read_label(TRAINING_LABEL_PATH, mnist::set_t::TRAINING));
    unique_ptr<vector<int>> testing_label(mnist::read_label(TESTING_LABEL_PATH, mnist::set_t::TESTING));
    mnist::normalize(training_image.get());
    mnist::normalize(testing_image.get());

    softmax::batch::init_arg args;
    args.weight_decay_rate = 1e-4;
    args.dim_input = mnist::IMAGE_ROWSIZE * mnist::IMAGE_COLSIZE;
    args.dim_output = mnist::NUM_LABEL_TYPE;
    args.num_training_data = mnist::NUM_TRAINING_IMAGE;
    args.num_testing_data = mnist::NUM_TESTING_IMAGE;
    args.max_epoch = 100;
    
    softmax::batch::classifier sm(args);
    sm.train(*training_image, *training_label, testing_image.get(), testing_label.get());
}

} // namespace batch

namespace mini_batch
{

classifier::classifier(const init_arg &args)
{
    check_arg(args);
    set_arg(args);
    arma::arma_rng::set_seed_random();

    field_args = make_unique<mf_t>(2, 1);
    (*field_args)(0, 0) = 0.005 * mat_t(DIM_OUTPUT, DIM_INPUT, arma::fill::randu);
    (*field_args)(1, 0) = 0.005 * vec_t(DIM_OUTPUT, arma::fill::randu);
    set_field_ptr(*field_args);
    output = make_unique<mat_t>(DIM_OUTPUT, BATCH_SIZE, arma::fill::zeros);
    prv_residue = make_unique<mat_t>(DIM_INPUT, BATCH_SIZE, arma::fill::zeros);

    optimization::grad_dsct::init_arg optm_args;
    optm_args.batch_size = BATCH_SIZE;
    optm = make_unique<optimization::wrapper<mf_t, optimization::grad_dsct::optimizer<mf_t>>>(optm_args, *field_args);
}

const mat_t *classifier::fp(const mat_t &input, const int idx_begin, const bool full_size)
{
    if (full_size)
        *output = *weight * input;
    else
        *output = *weight * input.cols(idx_begin, idx_begin + BATCH_SIZE - 1);
    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        output->col(i) += *bias;
        output->col(i) -= output->col(i).max();
        output->col(i) = arma::exp(output->col(i));
        output->col(i) /= arma::accu(output->col(i));
    }

    return output.get();
}
    
double classifier::data_func(const mat_t &input, const vector<int> &target, mf_t *new_grad, const int idx_begin, const bool full_size, const bool need_obj)
{
    double obj = 0.0;
    fp(input, idx_begin, full_size);

    if (need_obj)
    {
        for (int i = 0; i < BATCH_SIZE; ++i)
            obj += std::log((*output)(target[i], i));
        obj /= -BATCH_SIZE;
        if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
            obj += WEIGHT_DECAY_RATE / 2 * arma::accu(arma::pow(*weight, 2));
    }

    if (new_grad != nullptr)
    {
        const mat_t *output_residue = comp_output_residue(target, idx_begin);
        mat_t *new_w_grad = &(*new_grad)(0, 0);
        mat_t *new_b_grad = &(*new_grad)(1, 0);
        if (full_size)
            *new_w_grad = *output_residue * input.t() / BATCH_SIZE;
        else
            *new_w_grad = *output_residue * input.cols(idx_begin, idx_begin + BATCH_SIZE - 1).t() / BATCH_SIZE;
        *new_b_grad = arma::sum(*output_residue, 1) / BATCH_SIZE;
    
        if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
            *new_w_grad += WEIGHT_DECAY_RATE * *weight;
    }

    return obj;
}

double classifier::testing(const mat_t &input, const vector<int> &target, const bool need_print)
{
    if (target.size() % BATCH_SIZE != 0)
        CRY();

    int num_correct = 0;
    for (int i = 0; i < target.size(); i += BATCH_SIZE)
    {
        const mat_t *final_output = fp(input, i, false);

        for (int idx_output = i; idx_output < i + BATCH_SIZE; ++idx_output)
        {
            arma::uword idx_max;
            final_output->col(idx_output - i).max(idx_max);
            if (idx_max == target[idx_output])
                ++num_correct;
        }
    }
    
    double accuracy = static_cast<double>(num_correct) / target.size();
    if (need_print)
        cout << "testing accuracy: " << accuracy << endl;

    return accuracy;
}

void mini_batch_softmax_test()
{
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/mnist/train-images-idx3-ubyte"};
    const string TRAINING_LABEL_PATH{"/Users/evensong/ml_data/mnist/train-labels-idx1-ubyte"};
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/mnist/t10k-images-idx3-ubyte"};
    const string TESTING_LABEL_PATH{"/Users/evensong/ml_data/mnist/t10k-labels-idx1-ubyte"};

    unique_ptr<mat_t> training_image(mnist::read_vectorized_image(TRAINING_DATA_PATH, mnist::set_t::TRAINING));
    unique_ptr<mat_t> testing_image(mnist::read_vectorized_image(TESTING_DATA_PATH, mnist::set_t::TESTING));
    unique_ptr<vector<int>> training_label(mnist::read_label(TRAINING_LABEL_PATH, mnist::set_t::TRAINING));
    unique_ptr<vector<int>> testing_label(mnist::read_label(TESTING_LABEL_PATH, mnist::set_t::TESTING));
    mnist::normalize(training_image.get());
    mnist::normalize(testing_image.get());

    softmax::mini_batch::init_arg args;
    args.weight_decay_rate = 1e-4;
    args.dim_input = mnist::IMAGE_ROWSIZE * mnist::IMAGE_COLSIZE;
    args.dim_output = mnist::NUM_LABEL_TYPE;
    args.batch_size = 1;
    
    softmax::mini_batch::classifier sm(args);
    sm.train(*training_image, *training_label, testing_image.get(), testing_label.get());
}

} // namespace mini_batch

} // namespace softmax

namespace knn
{

classifier::classifier(const init_arg &args) 
{ 
    check_arg(args);
    set_arg(args);

    training_output = make_unique<vec_t>(NUM_TRAINING_DATA, arma::fill::zeros);
    testing_output = make_unique<vec_t>(NUM_TESTING_DATA, arma::fill::zeros);
    dist = make_unique<vector<pair<int, double>>>(NUM_TRAINING_DATA - 1);
    neighbor_cnt = make_unique<ivec_t>(DIM_OUTPUT, arma::fill::zeros);
}

const mat_t *classifier::do_classify(const mat_t &input, const vector<int> &target)
{
    vec_t *final_output = (target.size() == NUM_TRAINING_DATA ? training_output.get() : testing_output.get());

    for (int i = 0; i < target.size(); ++i)
    {
        neighbor_cnt->zeros();

        for (int j = 0; j < i; ++j)
        {
            (*dist)[j].first = j;
            (*dist)[j].second = comp_distance(input, i, j);
        }
        for (int j = i + 1; j < target.size(); ++j)
        {
            (*dist)[j - 1].first = j;
            (*dist)[j - 1].second = comp_distance(input, i, j);
        }
        std::nth_element(dist->begin(), dist->begin() + NUM_NEIGHBOR, dist->begin() + target.size() - 1, 
                [](const pair<int, double> &lhs, const pair<int, double> &rhs){ return lhs.second < rhs.second; });

        for (int i = 0; i < NUM_NEIGHBOR; ++i)
        {
            const int result = target[(*dist)[i].first];
            ++(*neighbor_cnt)[result];
        }
        arma::uword idx_max;
        neighbor_cnt->max(idx_max);
        
        (*final_output)[i] = idx_max;
    }

    return final_output;
}

double classifier::testing(const mat_t &input, const vector<int> &target, const bool is_training_data, const bool need_print)
{
    int num_input;
    if (is_training_data)
        num_input = NUM_TRAINING_DATA;
    else
        num_input = NUM_TESTING_DATA;
    const mat_t *final_output = do_classify(input, target);
    int num_correct = 0;
    for (int idx_output = 0; idx_output < num_input; ++idx_output)
    {
        if ((*final_output)[idx_output] == target[idx_output])
            ++num_correct;
    }
    
    double accuracy = static_cast<double>(num_correct) / num_input;
    if (need_print)
    {
        if (is_training_data)
            cout << "training_data ";
        else
            cout << "testing_data ";
        cout << "testing accuracy: " << accuracy << endl;
    }

    return accuracy;
}

void knn_test()
{
    const string ATTR_PATH{"/Users/evensong/ml_data/uci/iris/attr_name"};
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/uci/iris/training_data"};

    unique_ptr<uci::attr_info> attr(uci::read_attr(ATTR_PATH));
    unique_ptr<uci::classification_data> training_data(uci::read_data(TRAINING_DATA_PATH, *attr));
    uci::normalize(training_data.get());

    knn::init_arg knn_args;
    knn_args.num_neighbor = 5;
    knn_args.dim_output = attr->dim_output;
    knn_args.num_training_data = training_data->input.n_cols;
    knn_args.num_testing_data = training_data->input.n_cols;

    knn::classifier clsf(knn_args);
    clsf.testing(training_data->input, training_data->output, false, true);
}

} // namespace knn

#ifdef UNIT_TESTING
    
int main()
{
    knn::knn_test(); 
    return 0;
}

#endif

