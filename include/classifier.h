#ifndef EVENSONG_CLASSIFIER
#define EVENSONG_CLASSIFIER

#include "./stdafx.h"
#include "./field_algorithm.h"
#include "./prepare_data.h"
#include "./activation_function.h"
#include "./optimization.h"

namespace softmax
{

using namespace field_algorithm;

struct basic_init_arg
{
    double weight_decay_rate = 0.0;
    int dim_input = 0;
    int dim_output = 0;
};

namespace batch
{

struct init_arg : public basic_init_arg
{
    int max_epoch = 0;
    int num_training_data = 0;
    int num_testing_data = 0;
};

class classifier
{
    public :
        classifier(const init_arg &args);

        void train(const mat_t &input, const vector<int> &target,
                const mat_t *testing_input = nullptr, const vector<int> *testing_target = nullptr)
        {
            auto set_ptr_func_handler = [this](const mf_t &f){ set_field_ptr(f); };
            auto data_func_handler = [this, &input, &target](mf_t *new_grad){ 
                return data_func(input, target, new_grad); };
            if (testing_input != nullptr)
            {
                auto extra_func_handler = [this, testing_input, testing_target](const int epoch){ 
                    //cout << "epoch: " << epoch + 1 << '\t';
                    testing(*testing_input, *testing_target, false, true); 
                };
                optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler);
            }
            else
            {
                auto extra_func_handler = [](const int epoch){ /*cout << "epoch: " << epoch + 1 << endl;*/ };
                optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler);
            }
        }

        const mat_t *predict(const mat_t &input) { return fp(input, NUM_TRAINING_DATA); }

        const mat_t *tune(const mat_t &input, const vector<int> &target, mf_t *new_grad)
        {
            data_func(input, target, new_grad, false);
            return get_prv_residue();
        }

        double testing(const mat_t &input, const vector<int> &target, const bool is_training_data, const bool need_print);

        void name() { cout << "softmax\n"; }

        void set_field_ptr(const mf_t &f) 
        {
            weight = &f(0, 0);
            bias = &f(1, 0);
        }

        void transfer_field(f_mf_t *dst, const int dst_row, const int dst_col)
        {
            (*dst)(dst_row, dst_col) = *field_args;
        }

        void set_weight_decay_rate(const double rate) { WEIGHT_DECAY_RATE = rate; }

        // passing nullptr to new_grad: don't compute gradient
        // passing false to need_obj: don't compute objective value
        double data_func(const mat_t &input, const vector<int> &target, mf_t *new_grad, const bool need_obj = true);

        const mat_t *get_prv_residue()
        {
            *prv_residue = weight->t() * *output;
            return prv_residue.get();
        }

    private :
        const mat_t *fp(const mat_t &input, const int num_input);

        const mat_t *comp_output_residue(const vector<int> &target)
        {
            for (int i = 0; i < NUM_TRAINING_DATA; ++i)
                (*output)(target[i], i) -= 1.0;

            return output.get();
        }

        void check_arg(const init_arg &args)
        {
            if (args.dim_input <= 0)
                CRY("invalid dim_input");
            if (args.dim_output <= 0)
                CRY("invalid dim_output");
            if (std::abs(args.weight_decay_rate - 0.0) < DBL_EPSILON)
                cout << "softmax::batch::classifier: weight_decay_rate == 0.0" << endl;
            if (args.num_training_data <= 0)
                CRY("invalid num_training_data");
            if (args.num_testing_data <= 0)
                CRY("invalid num_testing_data");
            if (args.max_epoch <= 0)
                CRY("invalid max_epoch");
        }

        void set_arg(const init_arg &args)
        {
            WEIGHT_DECAY_RATE = args.weight_decay_rate;
            DIM_INPUT = args.dim_input;
            DIM_OUTPUT = args.dim_output;
            MAX_EPOCH = args.max_epoch;
            NUM_TRAINING_DATA = args.num_training_data;
            NUM_TESTING_DATA = args.num_testing_data;
        }

        friend void batch_softmax_test();

        unique_ptr<mf_t> field_args;
        const mat_t *weight;
        const mat_t *bias;
        unique_ptr<mat_t> output;
        unique_ptr<mat_t> testing_output;
        unique_ptr<mat_t> prv_residue;
        unique_ptr<optimization::wrapper<mf_t, optimization::l_bfgs::optimizer<mf_t>>> optm;

        double WEIGHT_DECAY_RATE; 
        int DIM_INPUT;
        int DIM_OUTPUT;
        int NUM_TRAINING_DATA;
        int NUM_TESTING_DATA;
        int MAX_EPOCH;
};

} // namespace batch

namespace mini_batch
{

struct init_arg : public basic_init_arg
{
    int batch_size = 0;
};

class classifier
{
    public :
        classifier(const init_arg &args);

        void train(const mat_t &input, const vector<int> &target,
                const mat_t *testing_input = nullptr, const vector<int> *testing_target = nullptr)
        {
            int idx_begin = 0;
            auto set_ptr_func_handler = [this](const mf_t &f){ set_field_ptr(f); };
            auto data_func_handler = [&](mf_t *new_grad){ return data_func(input, target, new_grad, idx_begin, false, false); };
            while (!optm->has_converged())
            {
                optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler);
                idx_begin += BATCH_SIZE;
                if (idx_begin >= input.n_cols)
                {
                    if (testing_input != nullptr)
                        testing(*testing_input, *testing_target); 
                    optm->alter_rate();
                    idx_begin = 0;
                }
            }
        }

        const mat_t *predict(const mat_t &input, const int idx_begin, const bool full_size) { return fp(input, idx_begin, full_size); }

        const mat_t *tune(const mat_t &input, const vector<int> &target, mf_t *new_grad, const int idx_begin)
        {
            data_func(input, target, new_grad, idx_begin, false);
            return get_prv_residue();
        }

        double testing(const mat_t &input, const vector<int> &target, const bool need_print = true);

        const mat_t *get_prv_residue()
        {
            *prv_residue = weight->t() * *output;
            return prv_residue.get();
        }

        void name() { cout << "softmax\n"; }

        void set_field_ptr(const mf_t &f) 
        {
            weight = &f(0, 0);
            bias = &f(1, 0);
        }

        void transfer_field(f_mf_t *dst, const int dst_row, const int dst_col)
        {
            (*dst)(dst_row, dst_col) = *field_args;
        }

        void set_weight_decay_rate(const double rate) { WEIGHT_DECAY_RATE = rate; }

        // passing nullptr to new_grad: don't compute gradient
        // passing false to need_obj: don't compute objective value
        double data_func(const mat_t &input, const vector<int> &target, mf_t *new_grad, const int idx_begin, const bool full_size, const bool need_obj = true);

    private :
        const mat_t *fp(const mat_t &input, const int idx_begin, const bool full_size);

        const mat_t *comp_output_residue(const vector<int> &target, const int idx_begin)
        {
            for (int i = 0; i < BATCH_SIZE; ++i)
                (*output)(target[i + idx_begin], i) -= 1.0;

            return output.get();
        }

        void check_arg(const init_arg &args)
        {
            if (args.dim_input <= 0)
                CRY("invalid dim_input");
            if (args.dim_output <= 0)
                CRY("invalid dim_output");
            if (std::abs(args.weight_decay_rate - 0.0) < DBL_EPSILON)
                cout << "softmax::batch_classifier: weight_decay_rate == 0.0" << endl;
            if (args.batch_size != 1 && args.batch_size != 2 && args.batch_size != 5 && args.batch_size != 10 && args.batch_size != 20 && args.batch_size != 50)
                CRY("batch_size is not supported");
        }

        void set_arg(const init_arg &args)
        {
            WEIGHT_DECAY_RATE = args.weight_decay_rate;
            DIM_INPUT = args.dim_input;
            DIM_OUTPUT = args.dim_output;
            BATCH_SIZE = args.batch_size;
        }

        friend void mini_batch_softmax_test();

        unique_ptr<mf_t> field_args;
        const mat_t *weight;
        const mat_t *bias;
        unique_ptr<mat_t> output;
        unique_ptr<mat_t> prv_residue;
        unique_ptr<optimization::wrapper<mf_t, optimization::grad_dsct::optimizer<mf_t>>> optm;

        double WEIGHT_DECAY_RATE; 
        int DIM_INPUT;
        int DIM_OUTPUT;
        int BATCH_SIZE; // 1 is best. the larger, the worse
};

} // namespace mini_batch

} // namespace softmax

namespace knn
{

struct init_arg
{
    int num_neighbor = 0;
    int dim_output = 0;
    int num_training_data = 0;
    int num_testing_data = 0;
};

class classifier
{
    public :
        classifier(const init_arg &args);

        void train(const mat_t &input, const vector<int> &target,
                const mat_t *testing_input = nullptr, const vector<int> *_testing_target = nullptr)
        {
            training_target = &target;
            testing_target = _testing_target;
        }

        const mat_t *predict(const mat_t &input) { return do_classify(input, *training_target); }

        const mat_t *tune(const mat_t &input, const vector<int> &target, mf_t *new_grad) { return nullptr; }

        double testing(const mat_t &input, const vector<int> &target, const bool is_training_data, const bool need_print);

        const mat_t *get_prv_residue() { CRY(); }

        void name() { cout << "knn\n"; }

        void set_field_ptr(const mf_t &f) { CRY(); }

        void transfer_field(f_mf_t *dst, const int dst_row, const int dst_col) { CRY(); }

        double data_func(const mat_t &input, const vector<int> &target, mf_t *new_grad, const bool need_obj = true) { CRY(); }

       
    private :

        const mat_t *do_classify(const mat_t &input, const vector<int> &target);

        double comp_distance(const mat_t &input, const int idx_a, const int idx_b) { return comp_euc_distance(input, idx_a, idx_b); }
        
        double comp_euc_distance(const mat_t &input, const int idx_a, const int idx_b)
        {
            return arma::norm(input.col(idx_a) - input.col(idx_b));
        }
        
        void check_arg(const init_arg &args)
        {
            if (args.num_neighbor <= 0)
                CRY();
            if (args.dim_output <= 0)
                CRY();
            if (args.num_training_data <= 0)
                CRY();
            if (args.num_testing_data <= 0)
                cout << "in knn: num_testing_data == 0" << endl;
        }

        void set_arg(const init_arg &args)
        {
            NUM_NEIGHBOR = args.num_neighbor;
            DIM_OUTPUT = args.dim_output;
            NUM_TRAINING_DATA = args.num_training_data;
            NUM_TESTING_DATA = args.num_testing_data;
        }

        friend void knn_test();

        unique_ptr<vec_t> training_output;
        unique_ptr<vec_t> testing_output;
        unique_ptr<vector<pair<int, double>>> dist;
        unique_ptr<ivec_t> neighbor_cnt;
        const vector<int> *training_target;
        const vector<int> *testing_target;

        int NUM_NEIGHBOR;
        int DIM_OUTPUT;
        int NUM_TRAINING_DATA;
        int NUM_TESTING_DATA;
};

} // namespace knn

#endif


