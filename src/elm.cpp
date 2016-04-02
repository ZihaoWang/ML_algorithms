#include <iostream>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "./prepare_data.h"
#include "/usr/local/Cellar/eigen/3.2.2/include/eigen3/Eigen/Core"
#include "/usr/local/Cellar/eigen/3.2.2/include/eigen3/Eigen/StdVector"
#include "/usr/local/Cellar/eigen/3.2.2/include/eigen3/Eigen/LU"

using std::cout;
using std::endl;

//#define USING_TESTING_SAMPLE

#ifdef USING_TESTING_SAMPLE
const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/regression/testing_sample"};
#else
const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/regression/sinc_sample"};
#endif
const string ATTRIBUTE_PATH{"/Users/evensong/ml_data/regression/sinc_attr_names"};
const string TESTING_DATA_PATH{"/Users/evensong/ml_data/regression/sinc_sample_test"};

class extreme_machine
{
    public:
        extreme_machine();
        void training(const int);
        void testing();

    private:
        static double act_func(const double x)
        {
            return std::exp(x * x / (-2));
        }

        int NUM_NUM_ATTRS = 0;
        int NUM_CTGR_ATTRS = 0;
        int NUM_ATTRS = 0;
        int NUM_EXAMPLES = 0;
        int NUM_TEST_EXAMPLES = 0;
        int DIM_INPUTS = 0;

        int NUM_HIDDEN_LAYER = 1;
        // temporarly
        int NUM_HIDDEN_LAYER_NODES = 20;
        int NUM_PAST_OUTPUTS = 0;

        shared_ptr<attr_info> attributes;    
        shared_ptr<sp_data> exms;
        shared_ptr<sp_data> test_exms;

        shared_ptr<vector<VectorXd, Eigen::aligned_allocator<VectorXd>>> hidden_weight;
        Eigen::MatrixXd input_weight;
        shared_ptr<vector<VectorXd, Eigen::aligned_allocator<VectorXd>>> bias;
};

void extreme_machine::training(const int num_layer)
{
    shared_ptr<Eigen::MatrixXd> input = std::make_shared<Eigen::MatrixXd>(exms->num_attr); // 以后修改这里
    shared_ptr<Eigen::MatrixXd> hidden_output = std::make_shared<Eigen::MatrixXd>((*input) * input_weight.transpose()); // row: exms; col: hidden nodes
    hidden_output->rowwise() += (*bias)[num_layer].transpose();
    auto activation = extreme_machine::act_func;
    *hidden_output = hidden_output->unaryExpr(activation);
    
    shared_ptr<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pseudo_inv = std::make_shared<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>((hidden_output->transpose() * (*hidden_output)).inverse() * hidden_output->transpose()); // 如果矩阵奇异，要改用SVD分解
    (*hidden_weight)[num_layer] = (*pseudo_inv) * exms->num_result;
}

void extreme_machine::testing()
{
    int num_layer = 0;

    shared_ptr<Eigen::MatrixXd> hidden_output = std::make_shared<Eigen::MatrixXd>(test_exms->num_attr * input_weight.transpose()); // row: exms; col: hidden nodes
    hidden_output->rowwise() += (*bias)[num_layer].transpose();
    auto activation = extreme_machine::act_func;
    *hidden_output = hidden_output->unaryExpr(activation);
    *hidden_output *= (*hidden_weight)[0];

    *hidden_output -= test_exms->num_result;
    *hidden_output = hidden_output->array().pow(2);
    double rmse = std::sqrt(hidden_output->sum() / NUM_TEST_EXAMPLES);
    
    cout << "test_rmse:\n" << rmse << endl;
}

int main()
{
    extreme_machine e;
    e.training(0);
    e.testing();
    return 0;
}

extreme_machine::extreme_machine() :
    attributes(std::make_shared<attr_info>()),
    exms(std::make_shared<sp_data>()),
    test_exms(std::make_shared<sp_data>()),
    hidden_weight(std::make_shared<vector<VectorXd, Eigen::aligned_allocator<VectorXd>>>()),
    bias(std::make_shared<vector<VectorXd, Eigen::aligned_allocator<VectorXd>>>())
{
    read_attr(ATTRIBUTE_PATH, attributes);
    NUM_NUM_ATTRS = attributes->num_attr_name.size();
    NUM_CTGR_ATTRS = attributes->ctgr_attr_name.size();
    NUM_ATTRS = NUM_NUM_ATTRS + NUM_CTGR_ATTRS;
    read_data(TRAINING_DATA_PATH, exms, NUM_NUM_ATTRS, NUM_CTGR_ATTRS, attributes->is_num_result);
    read_data(TESTING_DATA_PATH, test_exms, NUM_NUM_ATTRS, NUM_CTGR_ATTRS, attributes->is_num_result);
    NUM_EXAMPLES = exms->num_attr.rows();
    NUM_TEST_EXAMPLES = test_exms->num_attr.rows();
    DIM_INPUTS = NUM_NUM_ATTRS + NUM_PAST_OUTPUTS;

    // temporarly
    std::fill_n(std::back_inserter(*hidden_weight), NUM_HIDDEN_LAYER, VectorXd::Zero(NUM_HIDDEN_LAYER_NODES));
    input_weight = Eigen::MatrixXd::Random(NUM_HIDDEN_LAYER_NODES, DIM_INPUTS);
    std::fill_n(std::back_inserter(*bias), NUM_HIDDEN_LAYER, VectorXd::Random(NUM_HIDDEN_LAYER_NODES));
}





