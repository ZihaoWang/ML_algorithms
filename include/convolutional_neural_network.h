#ifndef EVENSONG_CNN
#define EVENSONG_CNN

#include "./stdafx.h"
#include "./prepare_data.h"
#include "./activation_function.h"
#include "./classifier.h"
#include "./field_algorithm.h"
#include "./optimization.h"

namespace cnn
{

struct init_arg
{
    int batch_size = 0;
    int num_conv_pooling_layer = 0;
    int num_fully_conv_layer = 0;
    int blank_edge_size = 0;

    double weight_decay_rate = 0.0;
    bool is_mean_pooling = false;

    vector<int> num_conv_ker;
    vector<int> conv_width;
    vector<int> pooling_width;
    unique_ptr<act_func::wrapper<act_func::hyp_tan>> act_function;
};

struct conv_ker
{
    conv_ker(const int conv_width, const int output_row, const int output_col, const int batch_size);

    void set_ptr(const mf_t &mf_now, const int row, const int col)
    {
        weight = &mf_now(row, col);
        bias = &mf_now(mf_now.n_rows - 1, 0)(row, col);
    }

    const mat_t *weight;
    const double *bias;
    unique_ptr<mat_t> rot_weight;
    cube_t result;
    cube_t result_drv;
    cube_t residue;
};

class model
{
    public :

        model(init_arg *args);

        void train(const cube_t &input, const vector<int> &target,
                const cube_t *testing_input = nullptr, const vector<int> *testing_target = nullptr);

        double testing(const cube_t &input, const vector<int> &target, const bool need_print = true);

        void distort_input(cube_t *input)
        {
            image_processor::distorter distrt(input->n_rows, input->n_cols);
            for (int i = 0; i < input->n_slices; ++i)
                distrt.process(&input->slice(i));
        }

        void set_field_ptr(const f_mf_t &f);

        double data_func(const cube_t &input, const vector<int> &target, f_mf_t *new_grad, const int idx_begin, const bool need_obj);

        void after_update_func();

    protected :

        void total_fp(const cube_t &input, const int idx_begin);

        void total_bp(const cube_t &input, const mat_t *backward_residue, f_mf_t *new_grad, const int idx_begin);

        void conv_pooling_fp(const cube_t &input, const int idx_begin, const int dst_layer);

        void conv_pooling_fp(const vector<cube_t> &input, const int dst_layer);

        void fully_conv_fp(const vector<cube_t> &input, const int dst_layer);

        void fill_final_input(const vector<cube_t> &input);

        void fill_clsf_residue(const mat_t *backward_residue);

        void fully_conv_bp(const vector<cube_t> &src_residue, const vector<cube_t> &forward_result, mf_t *new_grad, const int src_layer);

        void conv_pooling_bp(const vector<cube_t> &src_residue, vector<cube_t> *dst_residue, const vector<cube_t> &forward_result, mf_t *new_grad, const int src_layer);

        void conv_pooling_bp(const vector<cube_t> &src_residue, const cube_t &input, mf_t *new_grad, const int idx_begin);

        double WEIGHT_DECAY_RATE; 
        int BATCH_SIZE;
        int NUM_CONV_POOLING_LAYER;
        int NUM_FULLY_CONV_LAYER;
        int NUM_TOTAL_LAYER;
        int BLANK_EDGE_SIZE;
        bool IS_MEAN_POOLING;
        vector<int> NUM_CONV_KER;
        vector<int> CONV_WIDTH; 
        vector<int> POOLING_WIDTH;
        unique_ptr<act_func::wrapper<act_func::hyp_tan>> ACT_FUNCTION; 

        vector<int> conv_dst_row; // the sizes of pooling_residue {25, 10}
        vector<int> conv_dst_col;
        vector<int> pooling_dst_row; // the sizes of pooling_residue {25, 10}
        vector<int> pooling_dst_col;
        int final_input_row;
        int final_input_col;

        unique_ptr<vector<imat_t>> feature_conn;
        unique_ptr<f_mf_t> field_args; 
        unique_ptr<vector<vector<vector<unique_ptr<conv_ker>>>>> conv_map; 

        unique_ptr<vector<vector<cube_t>>> total_conv_result;
        unique_ptr<vector<vector<cube_t>>> forward_pooling;
        unique_ptr<vector<vector<cube_t>>> conv_residue;
        unique_ptr<vector<vector<cube_t>>> pooling_residue;

        unique_ptr<vector<vector<vector<conv_ker>>>> fully_conv_layer;
        unique_ptr<softmax::mini_batch::classifier> clsf;
        unique_ptr<optimization::wrapper<f_mf_t, optimization::grad_dsct::optimizer<f_mf_t>>> optm;

    private:
        friend void cnn_testing();

        unique_ptr<mat_t> final_input;
        unique_ptr<vector<cube_t>> clsf_residue;
        unique_ptr<vector<vector<vector<pair<int, int>>>>> pooling_pos;

        void max_pooling(const cube_t &src, cube_t *dst, const int idx_input,
                                  const int r_src, const int c_src, const int r_dst, const int c_dst, 
                                  const int r_offset, const int c_offset, const int idx_layer)
        {
            arma::uword r_max, c_max;
            (*dst)(r_dst, c_dst, idx_input) = src.slice(idx_input).submat(r_src, c_src, r_src + r_offset - 1, c_src + c_offset - 1).max(r_max, c_max);
            (*pooling_pos)[idx_layer][r_dst][c_dst].first = r_max;
            (*pooling_pos)[idx_layer][r_dst][c_dst].second = c_max;
        }
        
        void mean_pooling(const cube_t &src, cube_t *dst, const int idx_input,
                                  const int r_src, const int c_src, const int r_dst, const int c_dst,
                                  const int r_offset, const int c_offset)
        {
            double tmp = arma::accu(src.slice(idx_input).submat(r_src, c_src, r_src + r_offset - 1, c_src + c_offset - 1));
            (*dst)(r_dst, c_dst, idx_input) = tmp / (r_offset * c_offset);
        }
        
        void max_residue(const double residue, cube_t *dst, const int idx_input,
                         const int r_src, const int c_src, const int idx_layer)
        {
            const int r_dst = (*pooling_pos)[idx_layer][r_src][c_src].first;
            const int c_dst = (*pooling_pos)[idx_layer][r_src][c_src].second;
            (*dst)(r_dst, c_dst, idx_input) = residue;
        }
        
        void mean_residue(const double residue, cube_t *dst, const int idx_input,
                          const int r_dst, const int c_dst, const int r_offset, const int c_offset)
        {
            dst->slice(idx_input).submat(r_dst, c_dst, r_dst + r_offset - 1, c_dst + c_offset - 1) *= residue;
            dst->slice(idx_input).submat(r_dst, c_dst, r_dst + r_offset - 1, c_dst + c_offset - 1) /= r_offset * c_offset;
        }

        void conv(const cube_t &input, conv_ker *kernel, const int idx_begin, const int idx_layer, const int idx_dst);

        void pooling(const cube_t &conv_result, const int idx_begin, const int idx_layer, const int idx_dst);

        void get_pooling_residue(const cube_t &prev_residue, const int idx_layer, const int idx_dst);

        void get_conv_residue(const cube_t &src, cube_t *dst, conv_ker *kernel, const int idx_layer);

        void comp_grad(const cube_t &forward_result, conv_ker *kernel, mat_t *new_w_grad, double *new_b_grad, const int idx_layer);

        void rot180(mat_t *mat);

        void check_arg(const init_arg &args);

        void set_arg(init_arg *args);

        void alloc_space();

        void init_feature_conn();

        void init_fully_conv_layer(const init_arg &cnn_args, int *total_row, int *total_col);

        void init_classifier(const init_arg &cnn_args);

        void init_optimizer();

        void init_field_arg();

};

} // namespace cnn

#endif
