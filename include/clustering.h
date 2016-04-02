#ifndef EVENSONG_CLUSTERING
#define EVENSONG_CLUSTERING

#include "./stdafx.h"

namespace clustering
{

namespace k_means
{

} // namespace k_means

namespace cpcl
{

struct init_arg
{
    int max_epoch = 0;
    int num_cluster = 0;
    double penalty_rate = 0.0;
    int input_row = 0;
    int input_col = 0;
};

struct coor_center
{
    int idx;
    double dis;
};

class cluster
{
    public :
        cluster(const init_arg &args);

        void train(const mat_t &input);

        double get_obj_residue() const 
        { 
            return obj_residue; 
        }

        const mat_t *get_grad_residue() const 
        { 
            return grad_residue.get(); 
        }

        const vec_t &get_center() 
        { 
            return *center; 
        }
        
    private :
        int MAX_EPOCH;
        int NUM_CLUSTER;
        double PENALTY_RATE;
        int INPUT_ROW;
        int INPUT_COL;

        arma::uword idx_win;
        int total_time;
        int epoch_now;
        unique_ptr<vec_t> center;
        unique_ptr<ivec_t> winning_time;
        unique_ptr<vec_t> winning_freq;
        unique_ptr<vec_t> input_distance;
        unique_ptr<mat_t> grad_residue;
        double obj_residue;

        friend void testing();

        double update_center(const double input, const double min_dis, const bool need_residue);

        double comp_winner(const double input)
        {
            for (int i = 0; i < NUM_CLUSTER; ++i)
                (*input_distance)(i) = (*winning_freq)(i) * std::pow(input - (*center)(i), 2);
            return input_distance->min(idx_win);
        }

        void check_arg(const init_arg &args)
        {
            if (args.max_epoch <= 0)
                CRY();
            if (args.num_cluster <= 0)
                CRY();
            if (std::abs(args.penalty_rate - 0.0) < DBL_EPSILON)
                CRY();
            if (args.input_row <= 0)
                CRY();
            if (args.input_col <= 0)
                CRY();
        }

        void set_arg(const init_arg &args)
        {
            MAX_EPOCH = args.max_epoch;
            NUM_CLUSTER = args.num_cluster;
            PENALTY_RATE = args.penalty_rate;
            INPUT_ROW = args.input_row;
            INPUT_COL = args.input_col;
        }
};

} // namespace cpcl

namespace rpccl
{

struct init_arg
{
    int max_times = 0;
    int num_training_data = 0;
    int num_init_cluster = 0;
    int numeric_dim = 0;
    double penalty_rate = 0.0;
    bool need_residue = false;
    int input_row = 0; // for posterior
    int input_col = 0; // for posterior
};

class cluster
{
    public :
        cluster(const init_arg &args);

        void train(const mat_t &input);

        double get_obj_residue() const 
        { 
            if (!need_residue)
                CRY("has not prepared residue");
            return obj_residue; 
        }

        const mat_t *get_grad_residue() const 
        { 
            if (!need_residue)
                CRY("has not prepared residue");
            return grad_residue.get(); 
        }

        const vec_t &get_center() { return *center; }
        
        const ivec_t &get_num_owning() { return *num_owning; }

        const ivec_t &get_winning_time() { return *winning_time; }

        const vec_t &get_winning_freq() { return *winning_freq; }

        double get_num_cluster() const { return NUM_CLUSTER_NOW; }

        bool is_alive(const int idx_cluster) const { return cluster_life[idx_cluster]; }

        bool is_valid_data(const double d) const { return (std::abs(d - std::numeric_limits<double>::max()) < DBL_EPSILON) ? false : true; }

    private :

        void check_arg(const init_arg &args)
        {
            if (args.max_times <= 0)
                CRY();
            if (args.num_training_data <= 0)
                CRY();
            if (args.num_init_cluster <= 0)
                CRY();
            if (args.numeric_dim <= 0)
                CRY();
            if (args.numeric_dim > 1)
                CRY("not support");
            if (std::abs(args.penalty_rate - 0.0) < DBL_EPSILON)
                CRY();
            if (!args.need_residue)
                cout << "clustering without computing residue" << endl;
            if (args.input_row == 0)
                CRY();
            if (args.input_col == 0)
                CRY();
        }

        void set_arg(const init_arg &args)
        {
            MAX_TIMES = args.max_times;
            NUM_TRAINING_DATA = args.num_training_data;
            NUM_INIT_CLUSTER = args.num_init_cluster;
            NUMERIC_DIM = args.numeric_dim;
            PENALTY_RATE = args.penalty_rate;
            need_residue = args.need_residue;
            INPUT_ROW = args.input_row;
            INPUT_COL = args.input_col;
        }

        void comp_num_metric(const double input, bool comp_residue);

        void get_winner(const double input, const int row, const int col, bool comp_residue);

        void comp_idx_penalty();

        void update_center(const mat_t &input, const int row, const int col);

        friend void testing();

        vector<bool> cluster_life;
        unique_ptr<vec_t> center;
        unique_ptr<vec_t> numeric_distance;
        unique_ptr<ivec_t> num_owning;
        unique_ptr<imat_t> belonging;
        unique_ptr<vec_t> sum_belonging;
        unique_ptr<ivec_t> winning_time;
        unique_ptr<vec_t> winning_freq;
        unique_ptr<vec_t> penalty_weight;

        double obj_residue;
        unique_ptr<mat_t> grad_residue;
        vec_t grad_tmp;

        bool has_started;
        arma::uword idx_win;
        arma::uword idx_penalty;
        int total_epoch;
        bool need_residue;
        int MAX_TIMES;
        int NUM_TRAINING_DATA;
        int NUM_INIT_CLUSTER;
        int NUMERIC_DIM;
        double PENALTY_RATE;
        int NUM_CLUSTER_NOW;
        int INPUT_ROW;
        int INPUT_COL;
};

} // namespace rpccl

namespace new_rpccl
{

struct init_arg
{
    int max_times = 0;
    int num_training_data = 0;
    int num_init_cluster = 0;
    int numeric_dim = 0;
    double penalty_rate = 0.0;
    bool need_residue = false;
    int input_row = 0; // for posterior
    int input_col = 0; // for posterior
};

class cluster
{
    public :
        cluster(const init_arg &args);

        void train(const mat_t &input);

        double get_obj_residue() const 
        { 
            if (!need_residue)
                CRY("has not prepared residue");
            return obj_residue; 
        }

        const mat_t *get_grad_residue() const 
        { 
            if (!need_residue)
                CRY("has not prepared residue");
            return grad_residue.get(); 
        }

        const vec_t &get_center() 
        { 
            return *mu; 
        }
        
        void print_status() const
        {
            cout << "owning: " << num_owning->t();
            //cout << "relation: " << relation->t();
            cout << "center: " << mu->t();
            cout << "precision: " << lambda->t();
            //cout << "beta: " << beta->t();
            //cout << "a: " << a_gam->t();
            //cout << "b: " << b_gam->t() << endl;
        }

        const ivec_t &get_num_owning() { return *num_owning; }

        double get_num_cluster() const { return NUM_CLUSTER_NOW; }

        bool is_alive(const int idx_cluster) const { return cluster_life[idx_cluster]; }

        bool is_valid_data(const double d) const { return (std::abs(d - std::numeric_limits<double>::max()) < DBL_EPSILON) ? false : true; }

    private :

        void comp_total_gau(const double input)
        {
            for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            {
                if (is_alive(i))
                    (*gau_result)(i) = comp_gau(input, i);
                else
                    (*gau_result)(i) = std::numeric_limits<double>::max();
            }
        }

        double comp_gau(const double x, const int idx_gau)
        {
            const static double coeff = 1.0 / std::sqrt(2.0 * 3.1415926535897932384626433832795);
            double prec = (*lambda)(idx_gau);
            double tmp = std::pow((x - (*mu)(idx_gau)), 2) * prec;
            return coeff * std::sqrt(prec) * std::exp(-0.5 * tmp);
        }

        void check_arg(const init_arg &args)
        {
            if (args.max_times <= 0)
                CRY();
            if (args.num_training_data <= 0)
                CRY();
            if (args.num_init_cluster <= 0)
                CRY();
            if (args.numeric_dim <= 0)
                CRY();
            if (args.numeric_dim > 1)
                CRY("not support");
            if (std::abs(args.penalty_rate - 0.0) < DBL_EPSILON)
                cout << "penalty rate == 0" << endl;
            if (!args.need_residue)
                cout << "clustering without computing residue" << endl;
            if (args.input_row == 0)
                CRY();
            if (args.input_col == 0)
                CRY();
        }

        void set_arg(const init_arg &args)
        {
            MAX_TIMES = args.max_times;
            NUM_TRAINING_DATA = args.num_training_data;
            NUM_INIT_CLUSTER = args.num_init_cluster;
            NUMERIC_DIM = args.numeric_dim;
            PENALTY_RATE = args.penalty_rate;
            need_residue = args.need_residue;
            INPUT_ROW = args.input_row;
            INPUT_COL = args.input_col;
        }

        void comp_num_metric(const double input, const int row, const int col, bool comp_residue);

        void get_winner(const double input, const int row, const int col, bool comp_residue);

        void comp_idx_penalty();

        void update_arg(const double input);

        friend void testing();

        unique_ptr<vec_t> m_gau;
        unique_ptr<vec_t> beta;
        unique_ptr<vec_t> a_gam;
        unique_ptr<vec_t> b_gam;
        unique_ptr<vec_t> mu;
        unique_ptr<vec_t> lambda;
        unique_ptr<vec_t> pi;
        unique_ptr<vec_t> bar_input;
        unique_ptr<vec_t> gau_result;
        unique_ptr<vec_t> relation;

        unique_ptr<ivec_t> winning_time;
        unique_ptr<vec_t> winning_freq;
        vector<bool> cluster_life;
        unique_ptr<vec_t> numeric_distance;
        unique_ptr<ivec_t> num_owning;
        unique_ptr<vec_t> penalty_weight;

        double obj_residue;
        unique_ptr<mat_t> grad_residue;

        int total_epoch;
        bool has_started;
        arma::uword idx_win;
        arma::uword idx_penalty;
        bool need_residue;
        int MAX_TIMES;
        int NUM_TRAINING_DATA;
        int NUM_INIT_CLUSTER;
        int NUMERIC_DIM;
        double PENALTY_RATE;
        int NUM_CLUSTER_NOW;
        int INPUT_ROW;
        int INPUT_COL;
};

} // namespace new_rpccl

} // namespace clustering

#endif
