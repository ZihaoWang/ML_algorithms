#ifndef EVENSONG_SAMPLING
#define EVENSONG_SAMPLING

#include "./stdafx.h"

namespace hmc
{

struct init_arg
{
    int dim_input = 0;
    int num_burning = 10; // number of burning samples
    int num_skip = 40; // number of skipped samples for keeping independency
    int num_step = 20; // leapfrog epoch for each sample
    double step_size = 0.3; // size of leapfrog step
};

class model final
{
    public :

        model(const init_arg &args);

        const mat_t *run(const std::function<double(vec_t *, const vec_t &, const bool)> &data_func, const int num_sample);

    private :

        void leapfrog(const std::function<double(vec_t *, const vec_t &, const bool)> &data_func)
        {
            for (int i = 0; i < NUM_STEP; ++i)
            {
                data_func(position_drv.get(), *position, false);
                *momentum -= (STEP_SIZE / 2) * *position_drv;
                *position += STEP_SIZE * *momentum;
                data_func(position_drv.get(), *position, false);
                *momentum -= (STEP_SIZE / 2) * *position_drv;
            }
        }

        bool accept()
        {
            return (std::log(dist(gen)) < hamilton_old - hamilton) ? true : false;
        }

        void update_status()
        {
            *position_old = *position;
            obj_old = obj;
        }

        void check_arg(const init_arg &args);

        void set_arg(const init_arg &args);

        unique_ptr<mat_t> result;
        unique_ptr<vec_t> momentum;
        unique_ptr<vec_t> position_old;
        unique_ptr<vec_t> position;
        unique_ptr<vec_t> position_drv;
        double hamilton;
        double hamilton_old;
        double obj;
        double obj_old;
        std::default_random_engine gen;
        std::uniform_real_distribution<double> dist;

        int DIM_INPUT;
        int NUM_BURNING;
        int NUM_SKIP;
        int NUM_STEP;
        double STEP_SIZE;
};

} // namespace hmc

#endif
