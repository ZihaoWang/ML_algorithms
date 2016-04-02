#include "../include/convolutional_neural_network.h"

//#define DEBUG
#define UNIT_TESTING

namespace cnn
{

conv_ker::conv_ker(const int conv_width, const int output_row, const int output_col, const int batch_size) :
    weight(nullptr),
    bias(nullptr),
    rot_weight(new mat_t(conv_width, conv_width, arma::fill::zeros)),
    result(cube_t(output_row, output_col, batch_size, arma::fill::zeros)),
    result_drv(cube_t(output_row, output_col, batch_size, arma::fill::zeros)),
    residue(cube_t(output_row + 2 * (conv_width - 1), output_col + 2 * (conv_width - 1), batch_size, arma::fill::zeros))
{}

void model::train(const cube_t &input, const vector<int> &target, 
        const cube_t *testing_input, const vector<int> *testing_target)
{
    int idx_begin = 0;
    auto data_func_handler = [this, &input, &target, &idx_begin](f_mf_t *new_grad){
        return data_func(input, target, new_grad, idx_begin, false); };

    while (!optm->has_converged())
    {
        optm->optimize(field_args.get(), *this, data_func_handler);
        after_update_func();
        idx_begin += BATCH_SIZE;

        if (idx_begin % 10000 == 0)
            cout << "epoch: " << idx_begin << endl;
        if (idx_begin >= input.n_slices)
        {
            if (testing_input != nullptr)
                testing(*testing_input, *testing_target);
            optm->alter_rate();
            idx_begin = 0;
        }
    }
}

double model::data_func(const cube_t &input, const vector<int> &target, f_mf_t *new_grad, const int idx_begin, const bool need_obj)
{
    mf_t *new_clsf_grad = &(*new_grad)(0, NUM_TOTAL_LAYER);
    total_fp(input, idx_begin);
    double obj = clsf->data_func(*final_input, target, new_clsf_grad, idx_begin, true, need_obj);
    const mat_t *raw_residue = clsf->get_prv_residue();
    total_bp(input, raw_residue, new_grad, idx_begin);

    return obj;
}

double model::testing(const cube_t &input, const vector<int> &target, const bool need_print)
{
    if (target.size() % BATCH_SIZE != 0)
        CRY();

    int num_correct = 0;
    for (int i = 0; i < target.size(); i += BATCH_SIZE)
    {
        total_fp(input, i);
        const mat_t *final_output = clsf->predict(*final_input, 0, true);

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

void model::total_fp(const cube_t &input, const int idx_begin)
{
    for (auto &each_layer : *total_conv_result)
        for (auto &each_map : each_layer)
            each_map.zeros();

    conv_pooling_fp(input, idx_begin, 0);
    for (int idx_layer = 1; idx_layer < NUM_CONV_POOLING_LAYER; ++idx_layer)
        conv_pooling_fp((*forward_pooling)[idx_layer - 1], idx_layer);
    
    if (NUM_FULLY_CONV_LAYER > 0)
    {
        fully_conv_fp(forward_pooling->back(), 0);
        for (int idx_layer = 1; idx_layer < NUM_FULLY_CONV_LAYER; ++idx_layer)
            fully_conv_fp((*total_conv_result)[idx_layer - 1], idx_layer);

        fill_final_input(total_conv_result->back());
    }
    else
        fill_final_input(forward_pooling->back());
}

void model::total_bp(const cube_t &input, const mat_t *backward_residue, f_mf_t *new_grad, const int idx_begin)
{
    for (auto &each_layer : *conv_residue)
        for (auto &each_residue : each_layer)
            each_residue.zeros();
    for (auto &each_layer : *pooling_residue)
        for (auto &each_residue : each_layer)
            if (IS_MEAN_POOLING)
               each_residue.ones();
            else
               each_residue.zeros();
 
    fill_clsf_residue(backward_residue);
    if (NUM_FULLY_CONV_LAYER > 0)
    {
        if (NUM_FULLY_CONV_LAYER == 1)
            fully_conv_bp(*clsf_residue, forward_pooling->back(), &(*new_grad)(0, NUM_CONV_POOLING_LAYER), 0);
        else
        {
            fully_conv_bp(*clsf_residue, (*total_conv_result)[NUM_TOTAL_LAYER - 2], &(*new_grad)(0, NUM_TOTAL_LAYER - 1), NUM_FULLY_CONV_LAYER - 1);
    
            for (int idx_layer = NUM_TOTAL_LAYER - 2; idx_layer > NUM_CONV_POOLING_LAYER; --idx_layer)
                fully_conv_bp((*conv_residue)[idx_layer + 1], (*total_conv_result)[idx_layer - 1], &(*new_grad)(0, idx_layer), idx_layer - NUM_CONV_POOLING_LAYER);
            fully_conv_bp((*conv_residue)[NUM_CONV_POOLING_LAYER + 1], forward_pooling->back(), &(*new_grad)(0, NUM_CONV_POOLING_LAYER), 0);
        }

        conv_pooling_bp((*conv_residue)[NUM_CONV_POOLING_LAYER], &(*conv_residue)[NUM_CONV_POOLING_LAYER - 1], 
                (*forward_pooling)[NUM_CONV_POOLING_LAYER - 2], &(*new_grad)(0, NUM_CONV_POOLING_LAYER - 1), NUM_CONV_POOLING_LAYER - 1);
    }
    else
        conv_pooling_bp(*clsf_residue, &(*conv_residue)[NUM_CONV_POOLING_LAYER - 1], 
                (*forward_pooling)[NUM_CONV_POOLING_LAYER - 2], &(*new_grad)(0, NUM_CONV_POOLING_LAYER - 1), NUM_CONV_POOLING_LAYER - 1);

    for (int idx_layer = NUM_CONV_POOLING_LAYER - 2; idx_layer > 0; --idx_layer)
        conv_pooling_bp((*conv_residue)[idx_layer + 1], &(*conv_residue)[idx_layer], (*forward_pooling)[idx_layer - 1], &(*new_grad)(0, idx_layer), idx_layer);
    conv_pooling_bp((*conv_residue)[1], input, &(*new_grad)(0, 0), idx_begin);
}

void model::conv(const cube_t &input, conv_ker *kernel, const int idx_begin, const int idx_layer, const int idx_dst)
{
    for (int idx_input = 0; idx_input < BATCH_SIZE; ++idx_input)
        for (int row = 0; row < conv_dst_row[idx_layer]; ++row) 
            for (int col = 0; col < conv_dst_col[idx_layer]; ++col) 
            {
                double result = 0.0;
                for (int i = 0; i < CONV_WIDTH[idx_layer]; ++i)
                    for (int j = 0; j < CONV_WIDTH[idx_layer]; ++j)
                        result += input(i + row, j + col, idx_input + idx_begin) * (*kernel->weight)(i, j);
                result += *kernel->bias;
                result = ACT_FUNCTION->obj(result);
                kernel->result(row, col, idx_input) = result;
                kernel->result_drv(row, col, idx_input) = ACT_FUNCTION->drv_wrt_result(result);
            }
    (*total_conv_result)[idx_layer][idx_dst] += kernel->result;
}

void model::pooling(const cube_t &conv_result, const int idx_begin, const int idx_layer, const int idx_dst)
{
    int row_offset = POOLING_WIDTH[idx_layer], col_offset = POOLING_WIDTH[idx_layer];

    for (int idx_input = 0; idx_input < BATCH_SIZE; ++idx_input)
        for (int row = 0, p_row = 0; row < conv_dst_row[idx_layer]; row += row_offset, ++p_row)
        {
            row_offset = POOLING_WIDTH[idx_layer];
            if (conv_dst_row[idx_layer] - row < 2 * POOLING_WIDTH[idx_layer])
                row_offset = conv_dst_row[idx_layer] - row;
    
            for (int col = 0, p_col = 0; col < conv_dst_col[idx_layer]; col += col_offset, ++p_col) 
            {
                col_offset = POOLING_WIDTH[idx_layer];
                if (conv_dst_col[idx_layer] - col < 2 * POOLING_WIDTH[idx_layer])
                    col_offset = conv_dst_col[idx_layer] - col;
                if (IS_MEAN_POOLING)
                    mean_pooling((*total_conv_result)[idx_layer][idx_dst], &(*forward_pooling)[idx_layer][idx_dst], idx_input, 
                            row, col, p_row, p_col, row_offset, col_offset);
                else
                    max_pooling((*total_conv_result)[idx_layer][idx_dst], &(*forward_pooling)[idx_layer][idx_dst], idx_input, 
                            row, col, p_row, p_col, row_offset, col_offset, idx_layer);
            }
        }
}

void model::get_pooling_residue(const cube_t &prev_residue, const int idx_layer, const int idx_dst)
{
    int row_offset = POOLING_WIDTH[idx_layer], col_offset = POOLING_WIDTH[idx_layer];

    for (int idx_input = 0; idx_input < BATCH_SIZE; ++idx_input)
        for (int p_row = 0, row = 0; row < prev_residue.n_rows; p_row += row_offset, ++row)
        {
            row_offset = POOLING_WIDTH[idx_layer];
            if (conv_dst_row[idx_layer] - p_row < 2 * POOLING_WIDTH[idx_layer])
                row_offset = conv_dst_row[idx_layer] - p_row;
            for (int p_col = 0, col = 0; col < prev_residue.n_cols; p_col += col_offset, ++col)
            {
                col_offset = POOLING_WIDTH[idx_layer];
                if (conv_dst_col[idx_layer] - p_col < 2 * POOLING_WIDTH[idx_layer])
                    col_offset = conv_dst_col[idx_layer] - p_col;
                if (IS_MEAN_POOLING)
                    mean_residue(prev_residue.slice(idx_input)(row, col), &(*pooling_residue)[idx_layer][idx_dst], idx_input, 
                            p_row, p_col, row_offset, col_offset);
                else
                    max_residue(prev_residue.slice(idx_input)(row, col), &(*pooling_residue)[idx_layer][idx_dst], idx_input, 
                            row, col, idx_layer);
            }
        }
}

void model::get_conv_residue(const cube_t &src, cube_t *dst, conv_ker *kernel, const int idx_layer)
{
    kernel->residue.zeros();
    const int ker_size = CONV_WIDTH[idx_layer];
    const int blank_size = ker_size - 1;
    const int residue_row_end = blank_size + conv_dst_row[idx_layer] - 1;
    const int residue_col_end = blank_size + conv_dst_col[idx_layer] - 1;

    for (int idx_input = 0; idx_input < BATCH_SIZE; ++idx_input)
    {
        kernel->residue.slice(idx_input).submat(blank_size, blank_size, residue_row_end, residue_col_end) = 
            src.slice(idx_input) % kernel->result_drv.slice(idx_input);
    
        if (dst != nullptr)
            for (int i = 0; i < dst->n_rows; ++i) // full-convolution
                for (int j = 0; j < dst->n_cols; ++j)
                    (*dst)(i, j, idx_input) += arma::accu(kernel->residue.slice(idx_input).submat(i, j, i + ker_size - 1, j + ker_size - 1) % *kernel->rot_weight);
    }
}

void model::comp_grad(const cube_t &forward_result, conv_ker *kernel, mat_t *new_w_grad, double *new_b_grad, const int idx_layer)
{
    const int ker_size = CONV_WIDTH[idx_layer];
    const int blank_size = ker_size - 1;
    const int residue_row_end = blank_size + conv_dst_row[idx_layer] - 1;
    const int residue_col_end = blank_size + conv_dst_col[idx_layer] - 1;
    new_w_grad->zeros();
    *new_b_grad = 0.0;

    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        for (int row = 0; row < ker_size; ++row)
            for (int col = 0; col < ker_size; ++col)
            {
                double tmp = arma::accu(forward_result.slice(i).submat(row, col, row + conv_dst_row[idx_layer] - 1, col + conv_dst_col[idx_layer] - 1) %
                        kernel->residue.slice(i).submat(blank_size, blank_size, residue_row_end, residue_col_end));
                (*new_w_grad)(row, col) += tmp;
            }
    
        *new_b_grad += arma::accu(kernel->residue.slice(i).submat(blank_size, blank_size, residue_row_end, residue_col_end));
    }
    if (BATCH_SIZE > 1)
    {
        *new_w_grad /= BATCH_SIZE;
        *new_b_grad /= BATCH_SIZE;
    }

    if (std::abs(WEIGHT_DECAY_RATE - 0.0) > DBL_EPSILON)
        *new_w_grad += WEIGHT_DECAY_RATE * *kernel->weight;
}

void model::after_update_func()
{
    for (auto &layer : *conv_map)
        for (auto &feature_row : layer)
            for (auto &kernel : feature_row)
                if (kernel != nullptr)
                {
                    *kernel->rot_weight = *kernel->weight;
                    rot180(kernel->rot_weight.get());
                }

    for (auto &layer : *fully_conv_layer)
        for (auto &feature_row : layer)
            for (auto &kernel : feature_row)
            {
                *kernel.rot_weight = *kernel.weight;
                rot180(kernel.rot_weight.get());
            }
}

void model::conv_pooling_fp(const cube_t &input, const int idx_begin, const int dst_layer)
{
    const imat_t &feature = (*feature_conn)[dst_layer];
    for (int idx_dst = 0; idx_dst < feature.n_cols; ++idx_dst)
        if (feature(0, idx_dst) == 1)
            conv(input, (*conv_map)[dst_layer][0][idx_dst].get(), idx_begin, dst_layer, idx_dst);

    for (int i = 0; i < feature.n_cols; ++i)
        pooling((*total_conv_result)[dst_layer][i], idx_begin, dst_layer, i);
}

void model::conv_pooling_fp(const vector<cube_t> &input, const int dst_layer)
{
    const imat_t &feature = (*feature_conn)[dst_layer];
    for (int idx_src = 0; idx_src < feature.n_rows; ++idx_src)
        for (int idx_dst = 0; idx_dst < feature.n_cols; ++idx_dst)
            if (feature(idx_src, idx_dst) == 1)
                conv(input[idx_src], (*conv_map)[dst_layer][idx_src][idx_dst].get(), 0, dst_layer, idx_dst);

    for (int i = 0; i < feature.n_cols; ++i)
        pooling((*total_conv_result)[dst_layer][i], 0, dst_layer, i);
}

void model::fully_conv_fp(const vector<cube_t> &input, const int dst_layer)
{
    const int global_layer = dst_layer + NUM_CONV_POOLING_LAYER;
    for (int idx_src = 0; idx_src < NUM_CONV_KER[global_layer - 1]; ++idx_src)
        for (int idx_dst = 0; idx_dst < NUM_CONV_KER[global_layer]; ++idx_dst)
            conv(input[idx_src], &(*fully_conv_layer)[dst_layer][idx_src][idx_dst], 0, global_layer, idx_dst);
}

void model::fill_final_input(const vector<cube_t> &input)
{
    for (int i = 0; i < BATCH_SIZE; ++i)
        for (int j = 0, dst_offset = 0; j < NUM_CONV_KER.back(); dst_offset += total_conv_result->back()[j].size(), ++j)
            std::copy_n(input[j].begin_slice(i), total_conv_result->back()[j].size(), final_input->begin_col(i) + dst_offset);
}

void model::fill_clsf_residue(const mat_t *raw_residue)
{
    int offset_size;
    if (NUM_FULLY_CONV_LAYER == 0)
        offset_size = pooling_dst_row.back() * pooling_dst_col.back();
    else
        offset_size = conv_dst_row.back() * conv_dst_col.back();
    for (int i = 0; i < BATCH_SIZE; ++i)
        for (int j = 0, dst_offset = 0; j < NUM_CONV_KER.back(); dst_offset += offset_size, ++j)
        {
            //cout << i << '\t' << j << '\t' << dst_offset << endl;
            std::copy_n(raw_residue->begin_col(i) + dst_offset, offset_size, (*clsf_residue)[j].begin_slice(i));
        }
}

void model::fully_conv_bp(const vector<cube_t> &src_residue, const vector<cube_t> &forward_result, mf_t *new_grad, const int src_layer)
{
    const int global_layer = NUM_CONV_POOLING_LAYER + src_layer; 
    for (int idx_dst = 0; idx_dst < NUM_CONV_KER[global_layer - 1]; ++idx_dst)
        for (int idx_src = 0; idx_src < NUM_CONV_KER[global_layer]; ++idx_src)
        {
            conv_ker &map_now = (*fully_conv_layer)[src_layer][idx_dst][idx_src];
            get_conv_residue(src_residue[idx_src], &(*conv_residue)[global_layer][idx_dst], &map_now, global_layer);
            comp_grad(forward_result[idx_dst], &map_now, 
                    &(*new_grad)(idx_dst, idx_src), &(*new_grad)(new_grad->n_rows - 1, 0)(idx_dst, idx_src), global_layer);
        }
}

void model::conv_pooling_bp(const vector<cube_t> &src_residue, vector<cube_t> *dst_residue, const vector<cube_t> &forward_result, mf_t *new_grad, const int src_layer)
{
    for (int idx_src = 0; idx_src < NUM_CONV_KER[src_layer]; ++idx_src)
        get_pooling_residue(src_residue[idx_src], src_layer, idx_src);

    const imat_t &feature = (*feature_conn)[src_layer];
    for (int idx_dst = 0; idx_dst < feature.n_rows; ++idx_dst)
        for (int idx_src = 0; idx_src < feature.n_cols; ++idx_src)
        {
            unique_ptr<conv_ker> &map_now = (*conv_map)[src_layer][idx_dst][idx_src];
            if (map_now != nullptr)
            {
                get_conv_residue((*pooling_residue)[src_layer][idx_src], &(*dst_residue)[idx_dst], map_now.get(), src_layer);
                comp_grad(forward_result[idx_dst], map_now.get(), &(*new_grad)(idx_dst, idx_src), &(*new_grad)(feature.n_rows, 0)(idx_dst, idx_src), src_layer);
            }
        }
}

void model::conv_pooling_bp(const vector<cube_t> &src_residue, const cube_t &input, mf_t *new_grad, const int idx_begin)
{
    for (int idx_src = 0; idx_src < NUM_CONV_KER[0]; ++idx_src)
        get_pooling_residue(src_residue[idx_src], 0, idx_src);

    const imat_t &feature = feature_conn->front();
    for (int idx_src = 0; idx_src < feature.n_cols; ++idx_src)
    {
        unique_ptr<conv_ker> &map_now = conv_map->front()[0][idx_src];
        if (map_now != nullptr)
        {
            get_conv_residue(pooling_residue->front()[idx_src], nullptr, map_now.get(), 0);
            comp_grad(input.slices(idx_begin, idx_begin + BATCH_SIZE - 1), map_now.get(), &(*new_grad)(0, idx_src), &(*new_grad)(feature.n_rows, 0)(0, idx_src), 0);
        }
    }
}

model::model(init_arg *cnn_args)
{
    check_arg(*cnn_args);
    set_arg(cnn_args);
    NUM_TOTAL_LAYER = NUM_CONV_POOLING_LAYER + NUM_FULLY_CONV_LAYER;
    alloc_space();
    init_feature_conn();

    int total_row = mnist::IMAGE_ROWSIZE + 2 * BLANK_EDGE_SIZE, total_col = mnist::IMAGE_COLSIZE + 2 * BLANK_EDGE_SIZE;
    for (int idx_layer = 0; idx_layer < NUM_CONV_POOLING_LAYER; ++idx_layer)
    {
        if (idx_layer == 0)
        {
            (*conv_residue)[idx_layer].reserve(1);
            for (int idx_ker = 0; idx_ker < 1; ++idx_ker)
                (*conv_residue)[idx_layer].emplace_back(cube_t(total_row, total_col, BATCH_SIZE, arma::fill::zeros));
        }
        else
        {
            (*conv_residue)[idx_layer].reserve(NUM_CONV_KER[idx_layer - 1]);
            for (int idx_ker = 0; idx_ker < NUM_CONV_KER[idx_layer - 1]; ++idx_ker)
                (*conv_residue)[idx_layer].emplace_back(cube_t(total_row, total_col, BATCH_SIZE, arma::fill::zeros));
        }

        total_row -= (CONV_WIDTH[idx_layer] - 1);
        total_col -= (CONV_WIDTH[idx_layer] - 1);

        conv_dst_row.push_back(total_row);
        conv_dst_col.push_back(total_col);
        (*pooling_residue)[idx_layer].reserve(NUM_CONV_KER[idx_layer]);
        (*total_conv_result)[idx_layer].reserve(NUM_CONV_KER[idx_layer]);
        for (int idx_ker = 0; idx_ker < NUM_CONV_KER[idx_layer]; ++idx_ker)
        {
            (*total_conv_result)[idx_layer].emplace_back(cube_t(total_row, total_col, BATCH_SIZE, arma::fill::zeros));
            (*pooling_residue)[idx_layer].emplace_back(cube_t(total_row, total_col, BATCH_SIZE, arma::fill::zeros));
        }

        (*conv_map)[idx_layer].reserve((*feature_conn)[idx_layer].n_rows);
        for (int i = 0; i < (*feature_conn)[idx_layer].n_rows; ++i)
        {
            (*conv_map)[idx_layer].emplace_back();
            (*conv_map)[idx_layer][i].reserve((*feature_conn)[idx_layer].n_cols);
            for (int j = 0; j < (*feature_conn)[idx_layer].n_cols; ++j)
            {
                (*conv_map)[idx_layer][i].emplace_back();
                if ((*feature_conn)[idx_layer](i, j) != 0)
                    (*conv_map)[idx_layer][i][j] = make_unique<conv_ker>(CONV_WIDTH[idx_layer], total_row, total_col, BATCH_SIZE);
            }
        }

        total_row /= POOLING_WIDTH[idx_layer];
        total_col /= POOLING_WIDTH[idx_layer];
        pooling_dst_row.push_back(total_row);
        pooling_dst_col.push_back(total_col);
        (*forward_pooling)[idx_layer].reserve(NUM_CONV_KER[idx_layer]);
        for (int idx_ker = 0; idx_ker < NUM_CONV_KER[idx_layer]; ++idx_ker)
            (*forward_pooling)[idx_layer].emplace_back(cube_t(total_row, total_col, BATCH_SIZE, arma::fill::zeros));

        (*pooling_pos)[idx_layer].reserve(total_row);
        for (int i = 0; i < total_row; ++i)
        {
            (*pooling_pos)[idx_layer].emplace_back();
            (*pooling_pos)[idx_layer][i].reserve(total_col);
            for (int j = 0; j < total_col; ++j)
                (*pooling_pos)[idx_layer][i].emplace_back();
        }
    }

    if (NUM_FULLY_CONV_LAYER > 0)
        init_fully_conv_layer(*cnn_args, &total_row, &total_col);
    final_input_row = total_row;
    final_input_col = total_col;
    final_input = make_unique<mat_t>(NUM_CONV_KER.back() * final_input_row * final_input_col, BATCH_SIZE, arma::fill::zeros);
    for (int idx_ker = 0; idx_ker < NUM_CONV_KER.back(); ++idx_ker)
        clsf_residue->emplace_back(final_input_row, final_input_col, BATCH_SIZE, arma::fill::zeros);

    init_classifier(*cnn_args);
    init_field_arg();
    set_field_ptr(*field_args);
    init_optimizer();
    after_update_func();
}

void model::init_field_arg()
{
    field_args = make_unique<f_mf_t>(1, NUM_TOTAL_LAYER + 1);
    for (int i = 0; i < NUM_CONV_POOLING_LAYER; ++i)
    {
        const imat_t *feature = &(*feature_conn)[i];
        (*field_args)(0, i) = mf_t(feature->n_rows + 1, feature->n_cols);
        (*field_args)(0, i)(feature->n_rows, 0) = mat_t(feature->n_rows, feature->n_cols);

        for (int r = 0; r < feature->n_rows; ++r)
            for (int c = 0; c < feature->n_cols; ++c)
            {
                if ((*feature)(r, c) == 0)
                    (*field_args)(0, i)(feature->n_rows, 0)(r, c) = std::numeric_limits<double>::max();
                else
                {
                    (*field_args)(0, i)(r, c) = 0.005 * mat_t(CONV_WIDTH[i], CONV_WIDTH[i], arma::fill::randu);
                    (*field_args)(0, i)(feature->n_rows, 0)(r, c) = 0.005 * arma::randu();
                }
            }
    }

    for (int i = NUM_CONV_POOLING_LAYER; i < NUM_TOTAL_LAYER; ++i)
    {
        const int num_prev_ker = NUM_CONV_KER[i - 1];
        const int num_now_ker = NUM_CONV_KER[i];
        (*field_args)(0, i) = mf_t(num_prev_ker + 1, num_now_ker);
        (*field_args)(0, i)(num_prev_ker, 0) = mat_t(num_prev_ker, num_now_ker);

        for (int r = 0; r < num_prev_ker; ++r)
            for (int c = 0; c < num_now_ker; ++c)
            {
                (*field_args)(0, i)(r, c) = 0.005 * mat_t(CONV_WIDTH[i], CONV_WIDTH[i], arma::fill::randu);
                (*field_args)(0, i)(num_prev_ker, 0)(r, c) = 0.005 * arma::randu();
            }
    }

    (*field_args)(0, NUM_TOTAL_LAYER) = mf_t(2, 1);
    (*field_args)(0, NUM_TOTAL_LAYER)(0, 0) = 0.005 * mat_t(mnist::NUM_LABEL_TYPE, NUM_CONV_KER.back() * final_input_row * final_input_col, arma::fill::randu);
    (*field_args)(0, NUM_TOTAL_LAYER)(1, 0) = 0.005 * vec_t(mnist::NUM_LABEL_TYPE, arma::fill::randu);
}

void model::init_fully_conv_layer(const init_arg &cnn_args, int *total_row, int *total_col)
{
    int idx_layer = NUM_CONV_POOLING_LAYER;

    for (auto &e : *fully_conv_layer)
    {
        (*conv_residue)[idx_layer].reserve(NUM_CONV_KER[idx_layer - 1]);
        for (int idx_ker = 0; idx_ker < NUM_CONV_KER[idx_layer - 1]; ++idx_ker)
            (*conv_residue)[idx_layer].emplace_back(*total_row, *total_col, BATCH_SIZE, arma::fill::zeros);

        *total_row -= CONV_WIDTH[idx_layer] - 1;
        *total_col -= CONV_WIDTH[idx_layer] - 1;
        conv_dst_row.push_back(*total_row);
        conv_dst_col.push_back(*total_col);

        (*total_conv_result)[idx_layer].reserve(NUM_CONV_KER[idx_layer]);
        for (int idx_ker = 0; idx_ker < NUM_CONV_KER[idx_layer]; ++idx_ker)
            (*total_conv_result)[idx_layer].emplace_back(*total_row, *total_col, BATCH_SIZE, arma::fill::zeros);

        e.reserve(NUM_CONV_KER[idx_layer - 1]);
        for (int idx_ker = 0; idx_ker < NUM_CONV_KER[idx_layer - 1]; ++idx_ker)
            e.emplace_back();
        for (int i = 0; i < NUM_CONV_KER[idx_layer - 1]; ++i)
        {
            e[i].reserve(NUM_CONV_KER[idx_layer]);
            for (int j = 0; j < NUM_CONV_KER[idx_layer]; ++j)
                e[i].emplace_back(CONV_WIDTH[idx_layer], *total_row, *total_col, BATCH_SIZE);
        }

        ++idx_layer;
    }
}

void model::init_classifier(const init_arg &cnn_args)
{
    softmax::mini_batch::init_arg clsf_args;
    clsf_args.weight_decay_rate = cnn_args.weight_decay_rate;
    clsf_args.dim_input = NUM_CONV_KER.back() * final_input_row * final_input_col;
    clsf_args.dim_output = mnist::NUM_LABEL_TYPE;
    clsf_args.batch_size = cnn_args.batch_size;

    clsf = make_unique<softmax::mini_batch::classifier>(clsf_args);
}

void model::init_optimizer()
{
    optimization::grad_dsct::init_arg optm_args;
    optm_args.batch_size = BATCH_SIZE;
    optm = make_unique<optimization::wrapper<f_mf_t, optimization::grad_dsct::optimizer<f_mf_t>>>(optm_args, *field_args);
}

void model::alloc_space()
{
    total_conv_result = make_unique<vector<vector<cube_t>>>(NUM_CONV_POOLING_LAYER + NUM_FULLY_CONV_LAYER);
    conv_residue = make_unique<vector<vector<cube_t>>>(NUM_CONV_POOLING_LAYER + NUM_FULLY_CONV_LAYER);

    conv_map = make_unique<vector<vector<vector<unique_ptr<conv_ker>>>>>(NUM_CONV_POOLING_LAYER);
    forward_pooling = make_unique<vector<vector<cube_t>>>(NUM_CONV_POOLING_LAYER);
    pooling_residue = make_unique<vector<vector<cube_t>>>(NUM_CONV_POOLING_LAYER);
    pooling_pos = make_unique<vector<vector<vector<pair<int, int>>>>>(NUM_CONV_POOLING_LAYER);

    fully_conv_layer = make_unique<vector<vector<vector<conv_ker>>>>(NUM_FULLY_CONV_LAYER);

    feature_conn = make_unique<vector<imat_t>>();
    clsf_residue = make_unique<vector<cube_t>>();

    feature_conn->reserve(NUM_CONV_POOLING_LAYER);
    conv_dst_row.reserve(NUM_CONV_POOLING_LAYER + NUM_FULLY_CONV_LAYER);
    conv_dst_col.reserve(NUM_CONV_POOLING_LAYER + NUM_FULLY_CONV_LAYER);
    pooling_dst_row.reserve(NUM_CONV_POOLING_LAYER);
    pooling_dst_col.reserve(NUM_CONV_POOLING_LAYER);
}

void model::check_arg(const init_arg &args)
{
    if (args.batch_size <= 0)
        CRY();
    if (args.num_conv_pooling_layer <= 0)
        CRY();
    if (args.num_fully_conv_layer <= 0)
        cout << "number of fully convolutional layer is 0" << endl;
    if (args.blank_edge_size <= 0)
        cout << "has no blank edge in input image" << endl;
    if (std::abs(args.weight_decay_rate - 0.0) < DBL_EPSILON)
        cout << "weight decay rate is 0.0" << endl;
    check_vector(args.num_conv_ker, args.num_conv_pooling_layer + args.num_fully_conv_layer);
    check_vector(args.conv_width, args.num_conv_pooling_layer + args.num_fully_conv_layer);
    check_vector(args.pooling_width, args.num_conv_pooling_layer);
    if (args.act_function == nullptr)
        CRY();
}

void model::set_arg(init_arg *args)
{
    WEIGHT_DECAY_RATE = args->weight_decay_rate;
    BATCH_SIZE = args->batch_size;
    NUM_CONV_POOLING_LAYER = args->num_conv_pooling_layer;
    NUM_FULLY_CONV_LAYER = args->num_fully_conv_layer;
    BLANK_EDGE_SIZE = args->blank_edge_size;
    IS_MEAN_POOLING = args->is_mean_pooling;

    NUM_CONV_KER = args->num_conv_ker;
    CONV_WIDTH = args->conv_width;
    POOLING_WIDTH = args->pooling_width;
    ACT_FUNCTION = std::move(args->act_function);
}

void model::rot180(mat_t *mat)
{
        double *p = mat->begin();
        double *q = mat->end();
        --q;
        while (p < q)
            std::swap(*p++, *q--);
}

void model::set_field_ptr(const f_mf_t &f_now)
{
    for (int i = 0; i < NUM_CONV_POOLING_LAYER; ++i)
    {
        const mf_t &mf_now = f_now(0, i);
        for (int r = 0; r < mf_now.n_rows - 1; ++r)
            for (int c = 0; c < mf_now.n_cols; ++c)
            {
                unique_ptr<conv_ker> &map_now = (*conv_map)[i][r][c];
                if (map_now == nullptr)
                    continue;
                else
                    map_now->set_ptr(mf_now, r, c);
            }
    }

    for (int i = NUM_CONV_POOLING_LAYER; i < NUM_TOTAL_LAYER; ++i)
    {
        const mf_t &mf_now = f_now(0, i);
        for (int r = 0; r < mf_now.n_rows - 1; ++r)
            for (int c = 0; c < mf_now.n_cols; ++c)
                (*fully_conv_layer)[i - NUM_CONV_POOLING_LAYER][r][c].set_ptr(mf_now, r, c);
    }

    clsf->set_field_ptr(f_now(0, NUM_TOTAL_LAYER));
}

void model::init_feature_conn()
{
    feature_conn->push_back(imat_t(1, NUM_CONV_KER.front()));
    for (int idx_layer = 1; idx_layer < NUM_CONV_POOLING_LAYER; ++idx_layer)
        feature_conn->push_back(imat_t(NUM_CONV_KER[idx_layer - 1], NUM_CONV_KER[idx_layer], arma::fill::ones));

    // 以下为不可移植的代码，如果修改网络结构，必须修改这些代码

    // 1 means having connection; 0 means having no connection
    (*feature_conn)[0] = {1, 1, 1, 1, 1, 1};

    (*feature_conn)[1] = { 
        {1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
        {1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
        {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
        {0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
        {0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
        {0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1}
    };
        
    // partial connection is better than fully-connection, in both accuracy and time cost
    // some alternative feature connections:

    // 6 * 50
/*
    (*feature_conn)[1] = { 
        {1, 0, 0, 0, 1, 1,  1, 0, 0, 1, 0, 1,  1, 0, 1, 0, 0, 1,  1, 0,  1, 0, 0, 1, 1, 1,  1, 0, 1, 1, 1, 0,  1, 0, 1,  1, 0, 1, 1, 1, 1,  1, 0, 0,  1, 0, 1, 0, 0,  1},
        {1, 1, 0, 0, 0, 1,  1, 1, 0, 0, 1, 0,  1, 1, 0, 1, 0, 0,  0, 1,  1, 1, 0, 0, 1, 1,  0, 1, 0, 1, 1, 1,  1, 1, 0,  1, 1, 0, 1, 1, 1,  0, 1, 0,  0, 1, 0, 1, 0,  1},
        {1, 1, 1, 0, 0, 0,  0, 1, 1, 0, 0, 1,  0, 1, 1, 0, 1, 0,  1, 0,  1, 1, 1, 0, 0, 1,  1, 0, 1, 0, 1, 1,  0, 1, 1,  1, 1, 1, 0, 1, 1,  0, 0, 1,  0, 0, 1, 0, 1,  1},
        {0, 1, 1, 1, 0, 0,  1, 0, 1, 1, 0, 0,  0, 0, 1, 1, 0, 1,  0, 1,  1, 1, 1, 1, 0, 0,  1, 1, 0, 1, 0, 1,  1, 0, 1,  1, 1, 1, 1, 0, 1,  1, 0, 0,  0, 0, 0, 1, 0,  1},
        {0, 0, 1, 1, 1, 0,  0, 1, 0, 1, 1, 0,  1, 0, 0, 1, 1, 0,  1, 0,  0, 1, 1, 1, 1, 0,  1, 1, 1, 0, 1, 0,  1, 1, 0,  1, 1, 1, 1, 1, 0,  0, 1, 0,  1, 0, 0, 0, 1,  1},
        {0, 0, 0, 1, 1, 1,  0, 0, 1, 0, 1, 1,  0, 1, 0, 0, 1, 1,  0, 1,  0, 0, 1, 1, 1, 1,  0, 1, 1, 1, 0, 1,  0, 1, 1,  0, 1, 1, 1, 1, 1,  0, 0, 1,  0, 1, 0, 0, 0,  1}
    };
*/
}

void cnn_testing()
{
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/mnist/train-images-idx3-ubyte"};
    const string TRAINING_LABEL_PATH{"/Users/evensong/ml_data/mnist/train-labels-idx1-ubyte"};
    const string TESTING_DATA_PATH{"/Users/evensong/ml_data/mnist/t10k-images-idx3-ubyte"};
    const string TESTING_LABEL_PATH{"/Users/evensong/ml_data/mnist/t10k-labels-idx1-ubyte"};

    const int blank_size = 2;
    unique_ptr<cube_t> training_image(mnist::read_blank_edge_image(TRAINING_DATA_PATH, blank_size, mnist::set_t::TRAINING));
    unique_ptr<cube_t> testing_image(mnist::read_blank_edge_image(TESTING_DATA_PATH, blank_size, mnist::set_t::TESTING));
    unique_ptr<vector<int>> training_label(mnist::read_label(TRAINING_LABEL_PATH, mnist::set_t::TRAINING));
    unique_ptr<vector<int>> testing_label(mnist::read_label(TESTING_LABEL_PATH, mnist::set_t::TESTING));
    mnist::normalize(training_image.get());
    mnist::normalize(testing_image.get());

    cnn::init_arg args;

    args.batch_size = 1;
    args.num_conv_pooling_layer = 2;
    args.num_fully_conv_layer = 1;  
    args.blank_edge_size = blank_size; // initial image is 32 * 32

    args.weight_decay_rate = 0.0;
    args.is_mean_pooling = true;

    args.num_conv_ker = {6, 16, 120}; // 如果修改这里，必须修改init_feature_conn()
    args.conv_width = {5, 5, 5};
    args.pooling_width = {2, 2};
    args.act_function = make_unique<act_func::wrapper<act_func::hyp_tan>>();

    unique_ptr<cnn::model> network = make_unique<cnn::model>(&args);

#ifdef DEBUG
    cube_t tmp_image(32, 32, 1); 
    tmp_image.slice(0) = training_image->slice(0);
    vector<int> tmp_label(1);
    tmp_label[0] = (*training_label)[0];
    network->train(tmp_image, tmp_label, &tmp_image, &tmp_label);
#else
    network->train(*training_image, *training_label, testing_image.get(), testing_label.get());
#endif
}

} // namespace cnn

// configure: batch_size = 1, num_conv_pooling_layer = 2, num_fully_conv_layer = 1, blank_size = 2,
//              weight_decay_rate = 0.0, mean_pooling, 
//              num_conv_ker = {6, 16, 120}, conv_width = {5, 5, 5}, pooling_width = {2, 2},
//              activation function = hyp_tan
//              optimization:: grad_dsct
//
// result: 
// epoch = 1: 93.04%
// 2: 96.89%
// 3: 97.69%
// 4: 98.23%
// 5: 98.37%
// 6: 98.41%
// 7: 98.46%
// 8: 98.55%
// 9: 98.59%
// 10: 98.58%
// 11: 98.64%
// 12: 98.68%
// 13: 98.72%
// 14: 98.75%
// 15: 98.75%
// 16: 98.77%
// 17: 98.77%
// 18: 98.75%
// 19: 98.74%
// 1 hour, 23mins, 51seconds
//
#ifdef UNIT_TESTING
int main()
{
    cnn::cnn_testing();
    return 0;
}

#endif
