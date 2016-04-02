#ifndef EVENSONG_AE
#define EVENSONG_AE

#include "./stdafx.h"
#include "./prepare_data.h"
#include "./activation_function.h"
#include "./classifier.h"
#include "./clustering.h"
#include "./field_algorithm.h"
#include "./optimization.h"

namespace auto_encoder
{

struct batch_init_arg
{
    int num_training_data = 0;
    int num_testing_data = 0;
    int max_epoch = 0;
    int dim_input = 0;
    int dim_encode_layer = 0;
    double weight_decay_rate = 0.0;
    double tuning_weight_decay_rate = 0.0;
    //double sparsity_decay_rate = 0.0;
    //double sparsity_parameter = 0.0;
    unique_ptr<act_func::wrapper<act_func::hyp_tan>> act_function;
};

class batch_network 
{
    public :
        batch_network(batch_init_arg *args);

        void train(const mat_t &input)
        {
            auto set_ptr_func_handler = [this](const mf_t &f){ set_field_ptr(f); };
            auto data_func_handler = [this, input](mf_t *new_grad){ return data_func(input, new_grad); };
            auto extra_func_handler = [](const int epoch){ cout << "epoch: " << epoch + 1 << endl; };

            optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler);
        }

        const mat_t *predict(const mat_t &input)
        {
            return encode(input, NUM_TRAINING_DATA);
        }

        const mat_t *tune(const mat_t &input, const mat_t &residue, mf_t *new_grad)
        {
            return single_bp(input, residue, 0, new_grad);
        }

        const mat_t *testing(const mat_t &input)
        {
            return encode(input, NUM_TESTING_DATA);
        }

        void after_update_func() {}

        const int get_dim_output() const { return DIM_ENCODE_LAYER; }

        void set_field_ptr(const mf_t &f) 
        {
            weight[0] = &f(0, 0);
            bias[0] = &f(1, 0);
            weight[1] = &f(0, 1);
            bias[1] = &f(1, 1);
        }

        void set_encode_field_ptr(const mf_t &f) 
        {
            weight[0] = &f(0, 0);
            bias[0] = &f(1, 0);
            weight[1] = nullptr;
            bias[1] = nullptr;
        }

        void transfer_field(f_mf_t *dst, const int dst_row, const int dst_col)
        {
            if ((*dst)(dst_row, dst_col).empty())
                (*dst)(dst_row, dst_col) = mf_t(2, 1);
            else
                if ((*dst)(dst_row, dst_col).n_rows != 2 || (*dst)(dst_row, dst_col).n_cols != 1)
                    CRY("incommensurable field size");

            (*dst)(dst_row, dst_col)(0, 0) = (*field_args)(0, 0);
            (*dst)(dst_row, dst_col)(1, 0) = (*field_args)(1, 0);
        }

        void name() {}

        double data_func(const mat_t &input, mf_t *new_grad, const bool need_obj = true);

        void set_tuning_weight_decay_rate() { WEIGHT_DECAY_RATE = TUNING_WEIGHT_DECAY_RATE; }

        void check_grad(const mat_t &input);

    protected :

        double WEIGHT_DECAY_RATE;
        double TUNING_WEIGHT_DECAY_RATE;
        int DIM_INPUT;
        int MAX_EPOCH;
        int NUM_TRAINING_DATA;
        int NUM_TESTING_DATA;
        int DIM_ENCODE_LAYER;
        unique_ptr<mf_t> field_args;
        unique_ptr<optimization::wrapper<mf_t, optimization::l_bfgs::optimizer<mf_t>>> optm;
        vector<const mat_t *> weight; 
        vector<const mat_t *> bias;
        unique_ptr<vector<mat_t>> output;
        unique_ptr<vector<mat_t>> drv;
        unique_ptr<vector<mat_t>> residue;
        unique_ptr<mat_t> prv_residue;

        void init_field_arg()
        {
            (*field_args)(0, 0) = 1.0 * mat_t(DIM_ENCODE_LAYER, DIM_INPUT, arma::fill::randu);
            (*field_args)(1, 0) = 1.0 * vec_t(DIM_ENCODE_LAYER, arma::fill::randu); 
            (*field_args)(0, 1) = 1.0 * mat_t(DIM_INPUT, DIM_ENCODE_LAYER, arma::fill::randu);
            (*field_args)(1, 1) = 1.0 * vec_t(DIM_INPUT, arma::fill::randu); 
            (*field_args)(0, 0) = (*field_args)(0, 0) * 2 - 1.0;
            (*field_args)(1, 0) = (*field_args)(1, 0) * 2 - 1.0;
            (*field_args)(0, 1) = (*field_args)(0, 1) * 2 - 1.0;
            (*field_args)(1, 1) = (*field_args)(1, 1) * 2 - 1.0;
        }

        const mat_t *encode(const mat_t &input, const int num_input) { return single_fp(input, num_input, 0); }

        const mat_t *decode(const mat_t &input) { return single_fp(input, NUM_TRAINING_DATA, 1); }

        const mat_t *fully_fp(const mat_t &input) { return decode(*encode(input, NUM_TRAINING_DATA)); }

        const mat_t *single_bp(const mat_t &input, const mat_t &backward_residue, const int dst_layer, mf_t *new_grad);

        const mat_t *comp_output_residue(const mat_t &input)
        {
            (*residue)[1] = *fully_fp(input) - input;
            return &(*residue)[1];
        }

        void comp_grad(const mat_t &forward_result, const int dst_layer, mf_t *new_grad);

    private:
        friend void mnist_auto_encoder_test();

        friend void uci_auto_encoder_test();

        friend void uci_data_compress_test();

        void check_arg(const batch_init_arg &args);

        void set_arg(batch_init_arg *args);

        const mat_t *single_fp(const mat_t &input, const int num_input, const int dst_layer);

        unique_ptr<act_func::wrapper<act_func::hyp_tan>> ACT_FUNCTION;
        unique_ptr<mat_t> testing_output;
        //double SPARSITY_DECAY_RATE;
        //double SPARSITY_PARAMETER;

        //unique_ptr<vector<vec_t>> sp_term;
};

namespace sws_auto_encoder 
{

struct batch_init_arg
{
    auto_encoder::batch_init_arg ae_args;
    int num_gaussian = 0;
};

class batch_network : public auto_encoder::batch_network
{
    public :
        batch_network(sws_auto_encoder::batch_init_arg *args);

        void train(const mat_t &input)
        {
            auto set_ptr_func_handler = [this](const mf_t &f){ set_field_ptr(f); };
            auto gm_set_ptr_func_handler = [this](const mf_t &f){ gm_set_field_ptr(f); };
            auto data_func_handler = [this, &input](mf_t *new_grad){ return data_func(input, new_grad); };
            auto gm_data_func_handler = [this, &input](mf_t *new_grad){ return gm_data_func(input, new_grad); };
            auto extra_func_handler = [](const int epoch){ cout << "epoch: " << epoch + 1 << endl; };
            auto after_update_func_handler = [this, &gm_set_ptr_func_handler, &gm_data_func_handler](){ 
                gm_optm->optimize(gm_field_args.get(), gm_set_ptr_func_handler, gm_data_func_handler); };

            optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler, after_update_func_handler);
        }

        void gm_set_field_ptr(const mf_t &gm_f);

        double data_func(const mat_t &input, mf_t *new_grad, const bool need_obj = true);

        void check_grad();

        void name()
        {
            cout << "layer0: \n";
            for (double e : *mean[0])
                cout << e << ',';
            cout << endl;
            //cout << "layer1: \n";
            //for (double e : *mean[1])
            //    cout << e << ',';
            //cout << endl;
        }

    private :

        void check_arg(const batch_init_arg &args)
        {
            if (args.num_gaussian <= 0)
                CRY("invalid num_gaussian");
            if (std::abs(args.ae_args.weight_decay_rate - 0.0) < DBL_EPSILON)
                cout << "weight_decay_rate = 0.0 in soft weight sharing autoencoder" << endl;
        }

        void set_arg(const batch_init_arg &args)
        {
            NUM_GAUSSIAN = args.num_gaussian;
        }

        void gm_init_field_arg();

        const mat_t *single_bp(const mat_t &input, const mat_t &backward_residue, const int dst_layer, mf_t *new_grad);

        pair<double, const mat_t *> comp_obj(const mat_t &input);

        void comp_grad(const mat_t &forward_result, const int dst_layer, mf_t *new_grad);

        double gm_data_func(const mat_t &input, mf_t *new_grad, const bool need_obj = true);

        void gm_comp_grad(const int dst_layer, mf_t *new_grad);

        void comp_posterior(const int dst_layer);

        void comp_variance(const int dst_layer) { (*variance)[dst_layer] = arma::exp(*aux_variance[dst_layer]); }

        void comp_prior(const int dst_layer) 
        { 
            (*prior)[dst_layer] = arma::exp(*aux_prior[dst_layer]); 
            (*prior)[dst_layer] /= arma::accu((*prior)[dst_layer]); 
        }

        double minus_obj_weight_decay();

        void plus_grad_weight_decay(mat_t *w_grad, const int dst_layer);

        double comp_gaussian(const double x, const int dst_layer, const int idx_gaussian)
        {
            const static double coeff = 1.0 / std::sqrt(2.0 * 3.1415926535897932384626433832795);
            double var = (*variance)[dst_layer](idx_gaussian);
            double tmp = std::pow((x - (*mean[dst_layer])(idx_gaussian)), 2) / var;
            return coeff / std::sqrt(var) * std::exp(-0.5 * tmp);
        }

        friend void mnist_gaussian_ae_test();

        friend void uci_gaussian_ae_test();

        int NUM_GAUSSIAN; 

        vector<const mat_t *> mean;
        vector<const mat_t *> aux_variance;
        vector<const mat_t *> aux_prior;

        unique_ptr<vector<vec_t>> variance;
        unique_ptr<vector<vec_t>> prior;
        unique_ptr<vector<cube_t>> posterior;
        unique_ptr<mf_t> gm_field_args;
        unique_ptr<optimization::wrapper<mf_t, optimization::l_bfgs::optimizer<mf_t>>> gm_optm;
        // field_args: col1: weight, bias   col2: mean, aux_variance, aux_prior
};

} // namespace of sws_auto_encoder

namespace rp_auto_encoder 
{

struct batch_init_arg
{
    auto_encoder::batch_init_arg ae_args;
    int num_init_cluster = 0;
    bool init_field = false;
};

class batch_network : public auto_encoder::batch_network
{
    public :
        batch_network(rp_auto_encoder::batch_init_arg *args);

        void train(const mat_t &input)
        {
            auto set_ptr_func_handler = [this](const mf_t &f){ set_field_ptr(f); };
            auto data_func_handler = [this, &input](mf_t *new_grad){ return data_func(input, new_grad); };
            auto extra_func_handler = [this](const int epoch)
            { 
                cout << "epoch: " << epoch + 1 << endl; 
                if ((epoch + 1) % 20 == 0)
                    (*clus)[0].print_status();
                //(*clus)[1].print_status();
            };
            auto after_update_func_handler = [this](){ after_update_func(); };

            optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler, after_update_func_handler);
        }

        double data_func(const mat_t &input, mf_t *new_grad, const bool need_obj = true);

        void after_update_func() 
        { 
            (*clus)[0].train(*weight[0]); 
            (*clus)[1].train(*weight[1]); 
            /*
            cout << "own:" << (*clus)[0].get_num_owning().t();
            cout << "freq:" << (*clus)[0].get_winning_freq().t();
            cout << "time:" << (*clus)[0].get_winning_time().t();
            cout << "center:" << (*clus)[0].get_center().t() << endl;
            */
        }

        void check_grad();

        void name()
        {
            cout << "layer0 centers:\n";
            for (double e : (*clus)[0].get_center())
                cout << e << ',' << ' ';
            cout << endl << endl;
            cout << "layer1 centers:\n";
            for (double e : (*clus)[1].get_center())
                cout << e << ',' << ' ';
            cout << endl << endl;
        }

    private :

        void check_arg(const batch_init_arg &args)
        {
            if (!(!args.init_field/* && args.ae_args.init_fieldi*/))
                CRY("invalid init_field");
            if (args.num_init_cluster == 0)
                CRY();
            if (std::abs(args.ae_args.weight_decay_rate - 0.0) < DBL_EPSILON)
                CRY("invalid weight_decay_rate in soft weight sharing autoencoder");
        }

        void set_arg(const batch_init_arg &args) { NUM_INIT_CLUSTER = args.num_init_cluster; }

        const mat_t *single_bp(const mat_t &input, const mat_t &backward_residue, const int dst_layer, mf_t *new_grad);

        void comp_grad(const mat_t &forward_result, const int dst_layer, mf_t *new_grad);

        double comp_obj_weight_decay();

        void plus_grad_weight_decay(mat_t *w_grad, const int dst_layer);

        friend void mnist_rpccl_ae_test();

        friend void uci_rpccl_ae_test();

        unique_ptr<vector<clustering::new_rpccl::cluster>> clus;
        int NUM_INIT_CLUSTER;
};

} // namespace of rp_auto_encoder


template <typename SUB_AE_INIT_T, typename CLSF_INIT_T>
struct stacked_ae_init_arg
{
    int num_training_data = 0;
    int num_testing_data = 0;
    int num_layer = 0;
    int max_epoch = 0;
    bool is_simple_training = false;
    vector<SUB_AE_INIT_T> sub_ae_args;
    CLSF_INIT_T clsf_arg;
};

template <typename SUB_AE_T, typename CLSF_T>
class stacked_ae
{
    public :
        template <typename SUB_AE_INIT_T, typename CLSF_INIT_T>
        stacked_ae(stacked_ae_init_arg<SUB_AE_INIT_T, CLSF_INIT_T> *args);

        void train(const mat_t &input, const vector<int> &target,
                const mat_t *testing_input = nullptr, const vector<int> *testing_target = nullptr);

        void simple_train(const mat_t &input);

        // 优化：可以不用创建sub_ae和clsf的weight和bias，而是把指针指向stacked_ae的对应地址
        const mat_t *predict(const mat_t &input)
        {
            const mat_t *total_ae_result = total_ae_fp(input, NUM_TRAINING_DATA);
            return clsf->predict(*total_ae_result);
        }

        /*
        const mat_t *tune(const mat_t &input, const vector<int> &target, 
                f_mf_t *new_grad, const int start_row, const int start_col)
        {
            return data_func(input, target, new_grad, start_row, start_col);
        }
        */

        double testing(const mat_t &input, const vector<int> &target, const bool need_print)
        {
            const mat_t *final_output = total_ae_fp(input, NUM_TESTING_DATA);
            return clsf->testing(*final_output, target, false, need_print);
        }

        void set_field_ptr(const f_mf_t &f, const int start_row = 0, const int start_col = 0) 
        {
            int i = 0;
            for (; i < NUM_LAYER; ++i)
                (*sub_ae)[i].set_encode_field_ptr(f(start_row, start_col + i));
            clsf->set_field_ptr(f(start_row, start_col + i));
        }

        double data_func(const mat_t &input, const vector<int> &target, 
                f_mf_t *new_grad, const int start_row, const int start_col, const bool need_obj = true);

    private :
        template <typename SUB_AE_INIT_T, typename CLSF_INIT_T>
        void check_arg(const stacked_ae_init_arg<SUB_AE_INIT_T, CLSF_INIT_T> &args);

        template <typename SUB_AE_INIT_T, typename CLSF_INIT_T>
        void set_arg(const stacked_ae_init_arg<SUB_AE_INIT_T, CLSF_INIT_T> &args)
        {
            NUM_LAYER = args.num_layer;
            MAX_EPOCH = args.max_epoch;
            NUM_TRAINING_DATA = args.num_training_data;
            NUM_TESTING_DATA = args.num_testing_data;
            simple_training = args.is_simple_training;
        }

        const mat_t *total_ae_fp(const mat_t &input, const int num_input);

        const mat_t *total_ae_bp(const mat_t &input, const mat_t &backward_residue,
                f_mf_t *new_grad, const int start_row, const int start_col);

        void fetch_field()
        {
            for (int i = 0; i < NUM_LAYER; ++i)
                (*sub_ae)[i].transfer_field(field_args.get(), 0, i);
            if (!simple_training)
                clsf->transfer_field(field_args.get(), 0, NUM_LAYER);
        }

        int NUM_LAYER;
        int MAX_EPOCH;
        int NUM_TRAINING_DATA;
        int NUM_TESTING_DATA;
        bool simple_training;
        unique_ptr<vector<SUB_AE_T>> sub_ae;
        unique_ptr<CLSF_T> clsf;
        unique_ptr<f_mf_t> field_args;
        unique_ptr<optimization::wrapper<f_mf_t, optimization::l_bfgs::optimizer<f_mf_t>>> optm;
        vector<const mat_t *> ae_result;
};

template <typename SUB_AE_T, typename CLSF_T>
void stacked_ae<SUB_AE_T, CLSF_T>::train(const mat_t &input, const vector<int> &target, 
        const mat_t *testing_input, const vector<int> *testing_target)
{
    simple_train(input);
    
    cout << "start training classifier" << endl;
    const mat_t *last_ae_result = total_ae_fp(input, NUM_TRAINING_DATA);
    clsf->train(*last_ae_result, target);

    cout << "start fine tuning" << endl;
    fetch_field();
    set_field_ptr(*field_args);
    for (auto &e : *sub_ae)
        e.set_tuning_weight_decay_rate();

    auto set_ptr_func_handler = [this](const f_mf_t &f){ set_field_ptr(f); };
    auto data_func_handler = [this, &input, &target](f_mf_t *new_grad){ return data_func(input, target, new_grad, 0, 0); };
    if (testing_input != nullptr)
    {
        auto extra_func_handler = [this, testing_input, testing_target](const int epoch){ 
            cout << "epoch: " << epoch + 1 << '\t';
            testing(*testing_input, *testing_target, true); 
        };
        optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler);
    }
    else
    {
        auto extra_func_handler = [](const int epoch){ cout << "epoch: " << epoch + 1 << '\t'; };
        optm->optimize(field_args.get(), set_ptr_func_handler, data_func_handler, extra_func_handler);
    }
    testing(*testing_input, *testing_target, true);
    for (auto &e : *sub_ae)
        e.name();
}

template <typename SUB_AE_T, typename CLSF_T>
void stacked_ae<SUB_AE_T, CLSF_T>::simple_train(const mat_t &input)
{
    cout << "start training sub-autoencoder" << endl;
    cout << "autoencoder 1" << endl;
    (*sub_ae)[0].train(input);
    for (int now = 1; now < NUM_LAYER; ++now)
    {
        cout << "autoencoder " << now + 1 << endl;
        ae_result[0] = (*sub_ae)[0].predict(input);
        int bfr = 1;
        for (; bfr < now; ++bfr)
            ae_result[bfr] = (*sub_ae)[bfr].predict(*ae_result[bfr - 1]);
        (*sub_ae)[now].train(*ae_result[bfr - 1]);
    }
}

template <typename SUB_AE_T, typename CLSF_T>
double stacked_ae<SUB_AE_T, CLSF_T>::data_func(const mat_t &input, const vector<int> &target, 
        f_mf_t *new_grad, const int start_row, const int start_col, const bool need_obj)
{
    //可以优化
    const mat_t *last_ae_result = total_ae_fp(input, NUM_TRAINING_DATA);
    mf_t *new_clsf_grad = &(*new_grad)(start_row, start_col + NUM_LAYER);
    double obj = clsf->data_func(*last_ae_result, target, new_clsf_grad, need_obj);
    const mat_t *clsf_residue = clsf->get_prv_residue();
    total_ae_bp(input, *clsf_residue, new_grad, start_row, start_col);

    return obj;
}

template <typename SUB_AE_T, typename CLSF_T>
const mat_t *stacked_ae<SUB_AE_T, CLSF_T>::total_ae_fp(const mat_t &input, const int num_input)
{
    if (num_input == NUM_TRAINING_DATA)
        ae_result[0] = (*sub_ae)[0].predict(input);
    else if (num_input == NUM_TESTING_DATA)
        ae_result[0] = (*sub_ae)[0].testing(input);
    else
        CRY("invalid num_input");

    for (int i = 1; i < NUM_LAYER; ++i)
        if (num_input == NUM_TRAINING_DATA)
            ae_result[i] = (*sub_ae)[i].predict(*ae_result[i - 1]);
        else
            ae_result[i] = (*sub_ae)[i].testing(*ae_result[i - 1]);

    return ae_result.back();
}

template <typename SUB_AE_T, typename CLSF_T>
const mat_t *stacked_ae<SUB_AE_T, CLSF_T>::total_ae_bp(const mat_t &input, const mat_t &backward_residue,
        f_mf_t *new_grad, const int start_row, const int start_col)
{
    mf_t *new_ae_grad = nullptr;
    new_ae_grad = &(*new_grad)(start_row, start_col + NUM_LAYER - 1);
    if (NUM_LAYER == 1)
        return (*sub_ae)[0].tune(input, backward_residue, new_ae_grad);

    const mat_t *ae_residue = (*sub_ae)[NUM_LAYER - 1].tune(*ae_result[NUM_LAYER - 2], backward_residue, new_ae_grad);
    for (int i = NUM_LAYER - 2; i > 0; --i)
    {
        new_ae_grad = &(*new_grad)(start_row, start_col + i);
        ae_residue = (*sub_ae)[i].tune(*ae_result[i - 1], *ae_residue, new_ae_grad);
    }
    new_ae_grad = &(*new_grad)(start_row, start_col);
    return (*sub_ae)[0].tune(input, *ae_residue, new_ae_grad);
}

template <typename SUB_AE_T, typename CLSF_T>
template <typename SUB_AE_INIT_T, typename CLSF_INIT_T>
stacked_ae<SUB_AE_T, CLSF_T>::stacked_ae(stacked_ae_init_arg<SUB_AE_INIT_T, CLSF_INIT_T> *args)
{
    arma::arma_rng::set_seed_random();
    check_arg(*args);
    set_arg(*args);
    
    sub_ae = make_unique<vector<SUB_AE_T>>();
    clsf = make_unique<CLSF_T>(args->clsf_arg);
    field_args = make_unique<f_mf_t>(1, NUM_LAYER + 1);
    sub_ae->reserve(NUM_LAYER);
    ae_result.reserve(NUM_LAYER);
    for (int i = 0; i < NUM_LAYER; ++i)
    {
        sub_ae->emplace_back(&args->sub_ae_args[i]);
        ae_result.push_back(nullptr);
    }
    fetch_field();

    optimization::l_bfgs::init_arg optm_args;
    optm_args.max_epoch = MAX_EPOCH;
    optm = make_unique<optimization::wrapper<f_mf_t, optimization::l_bfgs::optimizer<f_mf_t>>>(optm_args, *field_args);
}

template <typename SUB_AE_T, typename CLSF_T>
template <typename SUB_AE_INIT_T, typename CLSF_INIT_T>
void stacked_ae<SUB_AE_T, CLSF_T>::check_arg(const stacked_ae_init_arg<SUB_AE_INIT_T, CLSF_INIT_T> &args)
{
    static_assert(std::is_same<SUB_AE_T, auto_encoder::batch_network>::value || 
            std::is_same<SUB_AE_T, sws_auto_encoder::batch_network>::value ||
            std::is_same<SUB_AE_T, rp_auto_encoder::batch_network>::value, 
            "in stacked_ae, wrong SUB_AE_T\n");
    static_assert(std::is_same<CLSF_T, softmax::batch::classifier>::value ||
            std::is_same<CLSF_T, knn::classifier>::value, 
            "in stacked_ae, wrong CLSF_T\n");
    if (args.num_layer <= 0)
        CRY("invalid num_layer");
    if (args.num_training_data <= 0)
        CRY("invalid num_training_data");
    if (args.num_testing_data <= 0)
        CRY("invalid num_testing_data");
    if (args.max_epoch <= 0)
        CRY("invalid max_epoch");
    if (args.sub_ae_args.size() != args.num_layer)
        CRY("invalid number of sub_ae_args");
}

} // namespace of auto_encoder

#endif
