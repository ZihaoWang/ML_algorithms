#include "../include/clustering.h"

//#define DEBUG
//#define UNIT_TESTING

namespace clustering
{

namespace new_rpccl
{

cluster::cluster(const init_arg &args)
{
    arma::arma_rng::set_seed_random();
    check_arg(args);
    set_arg(args);
    has_started = false;
    idx_win = std::numeric_limits<arma::uword>::max();
    idx_penalty = std::numeric_limits<arma::uword>::max();
    total_epoch = 0;
    NUM_CLUSTER_NOW = NUM_INIT_CLUSTER;

    m_gau = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    beta = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    a_gam = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    *a_gam *= NUM_TRAINING_DATA;
    b_gam = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    *b_gam *= NUM_TRAINING_DATA * 0.01;
    //*b_gam *= NUM_TRAINING_DATA * 0.01;
    mu = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::randu); 
    *mu = *mu * 1 - 0.5;
    lambda = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    pi = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    gau_result = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    relation = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);

    winning_time = make_unique<ivec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    winning_freq = make_unique<vec_t>(NUM_INIT_CLUSTER);
    winning_freq->fill(1.0 / NUM_CLUSTER_NOW);
    cluster_life.reserve(NUM_INIT_CLUSTER);
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        cluster_life.push_back(true);
    numeric_distance = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    num_owning = make_unique<ivec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    penalty_weight = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    if (need_residue)
    {
        obj_residue = 0.0;
        grad_residue = make_unique<mat_t>(args.input_row, args.input_col, arma::fill::zeros);
    }
}

void cluster::train(const mat_t &numeric_input)
{
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            (*winning_time)(i) = 1;
        else
            (*winning_time)(i) = 0;
    winning_freq->fill(1.0 / NUM_CLUSTER_NOW);
    if (numeric_input.n_rows != INPUT_ROW || numeric_input.n_cols != INPUT_COL)
        CRY();
    if (need_residue)
    {
        obj_residue = 0.0;
        grad_residue->zeros();
    }

    has_started = false;
    for (int idx_time = 0; idx_time < MAX_TIMES; ++idx_time)
    {
        num_owning->zeros(); // 必须每轮清零
        for (int i = 0; i < INPUT_ROW; ++i)
            for (int j = 0; j < INPUT_COL; ++j)
            {
                relation->zeros();
                //bar_input->zeros();
                comp_total_gau(numeric_input(i, j));
                if (idx_time < MAX_TIMES - 1)
                {
                    comp_num_metric(numeric_input(i, j), i, j, false);
                    get_winner(numeric_input(i, j), i, j, false);
                }
                else
                {
                    comp_num_metric(numeric_input(i, j), i, j, true);
                    get_winner(numeric_input(i, j), i, j, true);
                }
                update_arg(numeric_input(i, j));
#ifdef DEBUG
                cout << "m: " << m_gau->t();
                cout << "owning: " << num_owning->t();
                cout << "relation: " << relation->t();
                cout << "prec: " << lambda->t();
                cout << "beta: " << beta->t();
                cout << "a: " << a_gam->t();
                cout << "b: " << b_gam->t() << endl;
#endif
            }

        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if ((*num_owning)(i) == 0)
                if (is_alive(i))
                {           
                    (*pi)(i) = 0.0;
                    *pi /= arma::accu(*pi);
                    cluster_life[i] = false;
                    --NUM_CLUSTER_NOW;
                    cout << "remain: " << NUM_CLUSTER_NOW << endl;
                    print_status();
                    break;
                }
    }
}

void cluster::comp_num_metric(const double input, const int row, const int col, bool comp_residue)
{
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
    {
        if (!is_alive(i))
            (*numeric_distance)(i) = std::numeric_limits<double>::max();
        else
            (*numeric_distance)(i) = (*gau_result)(i);
    }

    double sum = 0.0;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            sum += (*numeric_distance)(i);
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
        {
            (*numeric_distance)(i) /= sum;
            (*relation)(i) = (*numeric_distance)(i);
            //(*bar_input)(i) = input;
        }
}

void cluster::get_winner(const double input, const int row, const int col, bool comp_residue)
{
    if (!has_started)
        numeric_distance->max(idx_win);
    else
    {
        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if (is_alive(i))
            {
                (*numeric_distance)(i) *= (*penalty_weight)(i);
                (*numeric_distance)(i) = 1.0 - (*numeric_distance)(i);
                (*numeric_distance)(i) *= (*winning_freq)(i);
            }
        numeric_distance->min(idx_win);
    }
    comp_idx_penalty();

    ++(*num_owning)(idx_win);
    ++total_epoch;
    (*winning_time)(idx_win) += 1;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            (*winning_freq)(i) = static_cast<double>((*winning_time)(i)) / total_epoch;

    (*penalty_weight)(idx_win) += PENALTY_RATE;
    if (is_alive(idx_penalty))
        (*penalty_weight)(idx_penalty) = std::max(0.0, (*penalty_weight)(idx_penalty) - PENALTY_RATE * (*numeric_distance)(idx_penalty));

    if (!has_started)
        has_started = true;

    if (need_residue && comp_residue)
    {
        vec_t tmp_post(NUM_INIT_CLUSTER, arma::fill::zeros);
        double term1 = 0.0;
        double term2 = 0.0;
        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if (is_alive(i))
            {
                tmp_post(i) = (*pi)(i) * (*gau_result)(i);
                term1 += (*lambda)(i) * (input - (*mu)(i)) * tmp_post(i);
            }
        double tmp_post_sum = arma::accu(tmp_post);
        term1 /= tmp_post_sum * tmp_post_sum;

        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if (is_alive(i))
            {
                obj_residue += (*numeric_distance)(i);
                term2 = -1 * (*lambda)(i) * (input - (*mu)(i)) / tmp_post_sum;
                (*grad_residue)(row, col) += (*winning_freq)(i) * (*penalty_weight)(i) * tmp_post(i) * (term1 + term2);
            }
        (*grad_residue)(row, col) *= -1;
    }
}

void cluster::update_arg(const double input)
{
    vec_t relation_var(NUM_INIT_CLUSTER, arma::fill::zeros);
    //for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
    //    relation_var(i) = (*relation)(i) * std::pow(input - (*bar_input)(i), 2);

    //double sum_pi = 0.0;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
    {
        if (is_alive(i))
        {
            //(*alpha)(i) += (*relation)(i);
            (*a_gam)(i) += 0.5 * (*relation)(i);
            (*b_gam)(i) += 0.5 * relation_var(i) + 
                    0.5 * (*beta)(i) * (*relation)(i) * std::pow(input - (*m_gau)(i), 2) / ((*beta)(i) + (*relation)(i));
            (*m_gau)(i) = ((*beta)(i) * (*m_gau)(i) + (*relation)(i) * input) / 
                    ((*beta)(i) + (*relation)(i));
            (*beta)(i) += (*relation)(i);

            //(*pi)(i) = (*alpha)(i) - 1.0;
            (*mu)(i) = (*m_gau)(i);
            (*lambda)(i) = ((*a_gam)(i) - 0.5) / (*b_gam)(i);
            //sum_pi += (*pi)(i);
        }
    }
    //*pi /= sum_pi;
}

void cluster::comp_idx_penalty()
{
    if (idx_win == 0)
        idx_penalty = 1;
    else
        idx_penalty = 0;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
    {
        if (i == idx_win)
            continue;
        if (has_started)
        {
            if ((*numeric_distance)(i) < (*numeric_distance)(idx_penalty))
                idx_penalty = i;
        }
        else
            if ((*numeric_distance)(i) > (*numeric_distance)(idx_penalty))
                idx_penalty = i;
    }
}

void testing()
{
    unique_ptr<mat_t> clustering_input = make_unique<mat_t>(1, 3000);
    std::default_random_engine gen1;
    std::default_random_engine gen2;
    std::default_random_engine gen3;
    std::normal_distribution<double> dis1(5.0, 1.0);
    std::normal_distribution<double> dis2(1.0, 1.0);
    std::normal_distribution<double> dis3(-4.0, 1.0);
    std::uniform_real_distribution<double> dis4(0, 1.0);
    /*
    for (int i = 0; i < 1000; ++i)
        (*clustering_input)(0, i) = dis1(gen1); 
    for (int i = 1000; i < 2000; ++i)
        (*clustering_input)(0, i) = dis2(gen2); 
    for (int i = 2000; i < 3000; ++i)
        (*clustering_input)(0, i) = dis3(gen3); 
        */
    for (int i = 0; i < 3000; ++i)
        (*clustering_input)(0, i) = dis4(gen1);
    std::random_shuffle(clustering_input->begin(), clustering_input->end());

    clustering::new_rpccl::init_arg clustering_args;
    clustering_args.max_times = 10;
    clustering_args.num_training_data = 3000;
    clustering_args.num_init_cluster = 4;
    clustering_args.numeric_dim = 1;
    clustering_args.penalty_rate = 1e-3;
    clustering_args.need_residue = true;
    clustering_args.input_row = 1;
    clustering_args.input_col = 3000;

    clustering::new_rpccl::cluster clus(clustering_args);

    clus.train(*clustering_input);
    cout << "center: " << clus.mu->t();
    cout << "precision: " << clus.lambda->t();
    //cout << "prior: " << clus.pi->t();
    cout << "obj: " << clus.get_obj_residue() << endl;
    cout << "life: ";
    for (auto e : *clus.num_owning)
        cout << e << '\t';
    cout << endl;
    for (int i = 0; i < clustering_args.num_init_cluster; ++i)
        if (clus.is_alive(i))
            cout << "live\t";
        else
            cout << "dead\t";
    cout << endl;
}

} // namespace new_rpccl

namespace rpccl
{

cluster::cluster(const init_arg &args)
{
    arma::arma_rng::set_seed_random();
    check_arg(args);
    set_arg(args);
    has_started = false;
    total_epoch = 0;
    idx_win = std::numeric_limits<arma::uword>::max();
    idx_penalty = std::numeric_limits<arma::uword>::max();
    NUM_CLUSTER_NOW = NUM_INIT_CLUSTER;

    cluster_life.reserve(NUM_INIT_CLUSTER);
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        cluster_life.push_back(true);
    center = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::randu); 
    numeric_distance = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    num_owning = make_unique<ivec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    belonging = make_unique<imat_t>(INPUT_ROW, INPUT_COL, arma::fill::zeros);
    sum_belonging = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::zeros);
    winning_time = make_unique<ivec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    winning_freq = make_unique<vec_t>(NUM_INIT_CLUSTER);
    winning_freq->fill(1.0 / NUM_CLUSTER_NOW);
    penalty_weight = make_unique<vec_t>(NUM_INIT_CLUSTER, arma::fill::ones);
    if (need_residue)
    {
        obj_residue = 0.0;
        grad_residue = make_unique<mat_t>(args.input_row, args.input_col, arma::fill::zeros);
        grad_tmp = vec_t(NUM_INIT_CLUSTER);
    }
    //*center *= 0.05;
}

void cluster::train(const mat_t &numeric_input)
{
    has_started = false;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            (*winning_time)(i) = 1;
        else
            (*winning_time)(i) = 0;
    winning_freq->fill(1.0 / NUM_CLUSTER_NOW);
    if (numeric_input.n_rows != INPUT_ROW || numeric_input.n_cols != INPUT_COL)
        CRY();
    if (need_residue)
    {
        obj_residue = 0.0;
        grad_residue->zeros();
    }

    for (int idx_time = 0; idx_time < MAX_TIMES; ++idx_time)
    {
        num_owning->zeros(); // 必须每轮清零
        sum_belonging->zeros();
        for (int i = 0; i < INPUT_ROW; ++i)
            for (int j = 0; j < INPUT_COL; ++j)
            {
                if (idx_time < MAX_TIMES - 1)
                {
                    comp_num_metric(numeric_input(i, j), false);
                    get_winner(numeric_input(i, j), i, j, false);
                }
                else
                {
                    comp_num_metric(numeric_input(i, j), true);
                    get_winner(numeric_input(i, j), i, j, true);
                }
                update_center(numeric_input, i, j);
            }

        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if ((*num_owning)(i) == 0)
                if (is_alive(i))
                {           
                    total_epoch -= (*winning_time)(i);
                    (*winning_time)(i) = 0;
                    (*winning_freq)(i) = 0.0;
                    cluster_life[i] = false;
                    --NUM_CLUSTER_NOW;
                    cout << "remain: " << NUM_CLUSTER_NOW << endl;
                    break;
                }
    }
    double sum = arma::accu(*winning_freq);
                    for (int j = 0; j < NUM_INIT_CLUSTER; ++j)
                        if (is_alive(j))
                            (*winning_freq)(j) /= sum;
}

void cluster::comp_num_metric(const double input, bool comp_residue)
{
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (!is_alive(i))
            (*numeric_distance)(i) = std::numeric_limits<double>::max();
        else
        {
            (*numeric_distance)(i) = std::abs(input - (*center)(i));
            (*numeric_distance)(i) *= -0.5;
            (*numeric_distance)(i) = std::exp((*numeric_distance)(i));
        }

    double sum = 0.0;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            sum += (*numeric_distance)(i);
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            (*numeric_distance)(i) /= sum;

    if (need_residue && comp_residue)
        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        {
            if (is_alive(i))
                grad_tmp(i) = 0.5 * ((*numeric_distance)(i) - (*numeric_distance)(i) * (*numeric_distance)(i));
            else
                grad_tmp(i) = std::numeric_limits<double>::max();
        }
}

void cluster::get_winner(const double input, const int row, const int col, bool comp_residue)
{
    if (!has_started)
        numeric_distance->min(idx_win);
    else
    {
        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if (is_alive(i))
            {
                (*numeric_distance)(i) *= (*penalty_weight)(i);
                (*numeric_distance)(i) = 1.0 - (*numeric_distance)(i);
                (*numeric_distance)(i) *= (*winning_freq)(i);
            }
        numeric_distance->min(idx_win);
    }
    comp_idx_penalty();

    (*belonging)(row, col) = idx_win;
    ++(*num_owning)(idx_win);
    ++total_epoch;
    (*winning_time)(idx_win) += 1;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
        if (is_alive(i))
            (*winning_freq)(i) = static_cast<double>((*winning_time)(i)) / total_epoch;

    (*penalty_weight)(idx_win) += PENALTY_RATE;
    if (is_alive(idx_penalty))
        (*penalty_weight)(idx_penalty) = std::max(0.0, (*penalty_weight)(idx_penalty) - PENALTY_RATE * (*numeric_distance)(idx_penalty));

    if (!has_started)
        has_started = true;

    if (need_residue && comp_residue)
    {
        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if (is_alive(i))
            {
                obj_residue += (*numeric_distance)(i);
                grad_tmp(i) *= (*penalty_weight)(i);
                grad_tmp(i) *= (*winning_freq)(i);
                double tmp = input - (*center)(i);
                if (std::abs(tmp - 0.0) < DBL_EPSILON)
                    continue;
                if (tmp < 0.0)
                    grad_tmp(i) *= -1;
            }

        for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
            if (is_alive(i))
                (*grad_residue)(row, col) += grad_tmp(i);
    }

}

void cluster::update_center(const mat_t &input, const int row, const int col)
{
    (*sum_belonging)((*belonging)(row, col)) += input(row, col);
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
    {
        if (is_alive(i))
        {
            if ((*num_owning)(i) != 0)
                (*center)(i) = (*sum_belonging)(i) / (*num_owning)(i);
        }
        else
            (*center)(i) = std::numeric_limits<double>::max();
    }

}

void cluster::comp_idx_penalty()
{
    if (idx_win == 0)
        idx_penalty = 1;
    else
        idx_penalty = 0;
    for (int i = 0; i < NUM_INIT_CLUSTER; ++i)
    {
        if (i == idx_win)
            continue;
        if ((*numeric_distance)(i) > (*numeric_distance)(idx_penalty))
            idx_penalty = i;
    }
}

void testing()
{
    unique_ptr<mat_t> clustering_input = make_unique<mat_t>(1, 3000);
    std::default_random_engine gen1;
    std::default_random_engine gen2;
    std::default_random_engine gen3;
    std::normal_distribution<double> dis1(5.0, 1.0);
    std::normal_distribution<double> dis2(1.0, 1.0);
    std::normal_distribution<double> dis3(-4.0, 1.0);
    for (int i = 0; i < 1000; ++i)
        (*clustering_input)(0, i) = dis1(gen1); 
    for (int i = 1000; i < 2000; ++i)
        (*clustering_input)(0, i) = dis2(gen2); 
    for (int i = 2000; i < 3000; ++i)
        (*clustering_input)(0, i) = dis3(gen3); 
    std::random_shuffle(clustering_input->begin(), clustering_input->end());

    clustering::rpccl::init_arg clustering_args;
    clustering_args.max_times = 20;
    clustering_args.num_training_data = 3000;
    clustering_args.num_init_cluster = 4;
    clustering_args.numeric_dim = 1;
    clustering_args.penalty_rate = static_cast<double>(clustering_args.num_init_cluster) / clustering_args.num_training_data;
    clustering_args.need_residue = true;
    clustering_args.input_row = 1;
    clustering_args.input_col = 3000;

    clustering::rpccl::cluster clus(clustering_args);

    clus.train(*clustering_input);
    cout << clus.center->t();
    cout << "obj: " << clus.get_obj_residue() << endl;
    cout << "life: ";
    for (auto e : *clus.num_owning)
        cout << e << '\t';
    cout << endl;
    for (int i = 0; i < clustering_args.num_init_cluster; ++i)
        if (clus.is_alive(i))
            cout << "live\t";
        else
            cout << "dead\t";
    cout << endl;
}

} // namespace rpccl

namespace cpcl
{

cluster::cluster(const init_arg &args)
{
    arma::arma_rng::set_seed_random();
    check_arg(args);
    set_arg(args);
    total_time = NUM_CLUSTER;
    idx_win = -1;
    epoch_now = -1;
    obj_residue = 0.0;

    center = make_unique<vec_t>(NUM_CLUSTER, arma::fill::randu);
    //*center *= 1.0;
    //*center = *center * 2 - 1.0;
    winning_time = make_unique<ivec_t>(NUM_CLUSTER, arma::fill::ones);
    winning_freq = make_unique<vec_t>(NUM_CLUSTER);
    input_distance = make_unique<vec_t>(NUM_CLUSTER, arma::fill::zeros);
    grad_residue = make_unique<mat_t>(INPUT_ROW, INPUT_COL, arma::fill::zeros);
    for (int i = 0; i < NUM_CLUSTER; ++i)
        (*winning_freq)(i) = (*winning_time)(i) / static_cast<double>(total_time);
}

void cluster::train(const mat_t &input)
{
    obj_residue = 0.0;
    for (epoch_now = 0; epoch_now < MAX_EPOCH - 1; ++epoch_now)
    {
        for (int i = 0; i < input.n_rows; ++i)
            for (int j = 0; j < input.n_cols; ++j)
            {
                const double each_input = input(i, j);
                const double min_dis = comp_winner(each_input);
                update_center(each_input, min_dis, false);
            }
    }
    for (int i = 0; i < input.n_rows; ++i)
        for (int j = 0; j < input.n_cols; ++j)
        {
            const double each_input = input(i, j);
            const double min_dis = comp_winner(each_input);
            (*grad_residue)(i, j) = update_center(each_input, min_dis, true);
        }
}

double cluster::update_center(const double input, const double min_dis, const bool need_residue)
{
    vector<coor_center> cluster_dis(NUM_CLUSTER);
    for (int i = 0; i < NUM_CLUSTER; ++i)
    {
        cluster_dis[i].idx = i;
        cluster_dis[i].dis = std::abs((*center)(i) - (*center)(idx_win));
    }
    std::sort(cluster_dis.begin(), cluster_dis.end(), [](const coor_center &a, const coor_center &b)->bool{ return a.dis < b.dis; });
    auto territory_end = std::find_if(cluster_dis.begin(), cluster_dis.end(), [min_dis](const coor_center &a)->bool{ return a.dis > min_dis; }); 
    const int num_coor = std::min(epoch_now + 1, static_cast<int>(territory_end - cluster_dis.begin()));

    double grad_res = 0.0;
    if (need_residue)
    {
        for (int i = 0; i < num_coor; ++i)
        {
            double tmp = input - (*center)(cluster_dis[i].idx);
            grad_res += (*winning_freq)[i] * tmp;
            obj_residue += (*winning_freq)[i] * tmp * tmp;
        }
        /*
            double tmp = input - (*center)(idx_win);
            grad_res += (*winning_freq)[idx_win] * tmp;
            obj_residue += (*winning_freq)[idx_win] * tmp * tmp;
        */
    }

    double dis_win = input - (*center)(cluster_dis[0].idx);
    (*center)(cluster_dis[0].idx) += PENALTY_RATE * dis_win;
    dis_win = std::abs(dis_win);

    std::for_each(cluster_dis.begin() + 1, cluster_dis.begin() + num_coor, [this, dis_win, input](const coor_center &a){
            double &center_now = (*center)(a.idx);
            center_now += PENALTY_RATE * dis_win / std::max(dis_win, std::abs(center_now - input)) * (input - center_now);
            });
    std::for_each(cluster_dis.begin() + num_coor, territory_end, [this, dis_win, input](const coor_center &a){
            double &center_now = (*center)(a.idx);
            center_now -= PENALTY_RATE * dis_win / std::abs(center_now - input) * (input - center_now);
            });

    (*winning_time)(idx_win) += 1;
    ++total_time;
    for (int i = 0; i < NUM_CLUSTER; ++i)
        (*winning_freq)(i) = (*winning_time)(i) / static_cast<double>(total_time);

    return grad_res * 2;
}

void testing()
{
    unique_ptr<mat_t> clustering_input = make_unique<mat_t>(1, 3000);
    std::default_random_engine gen1;
    std::default_random_engine gen2;
    std::default_random_engine gen3;
    std::normal_distribution<double> dis1(5.0, 1.0);
    std::normal_distribution<double> dis2(1.0, 1.0);
    std::normal_distribution<double> dis3(-4.0, 1.0);
    for (int i = 0; i < 1000; ++i)
        (*clustering_input)(0, i) = dis1(gen1); 
    for (int i = 1000; i < 2000; ++i)
        (*clustering_input)(0, i) = dis2(gen2); 
    for (int i = 2000; i < 3000; ++i)
        (*clustering_input)(0, i) = dis3(gen3); 
    std::random_shuffle(clustering_input->begin(), clustering_input->end());

    unique_ptr<mat_t> mimic_weight = make_unique<mat_t>(16, 200, arma::fill::randu);
    *mimic_weight *= 0.005;
    clustering::cpcl::init_arg clustering_args;
    clustering_args.max_epoch = 10;
    clustering_args.num_cluster = 10;
    clustering_args.penalty_rate = 1e-3;

    clustering::cpcl::cluster clus(clustering_args);

    clus.train(*mimic_weight);
    //cout << clus.center->t();
    /*
    cout << "obj: " << clus.get_obj_residue() << endl;
    cout << "life: ";
    for (auto e : *clus.num_owning)
        cout << e << '\t';
    cout << endl;
    for (int i = 0; i < clustering_args.num_init_cluster; ++i)
        if (clus.is_alive(i))
            cout << "live\t";
        else
            cout << "dead\t";
            */
    cout << endl;
}


} // namespace cpcl

} // namespace clustering

#ifdef UNIT_TESTING

int main()
{
    clustering::new_rpccl::testing();

    return 0;
}


#endif







