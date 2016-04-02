#include "../include/stdafx.h"
#include "../include/linear_search.h"
#include "../include/optimization.h"

#define UNIT_TESTING
#define VECTOR

#ifdef UNIT_TESTING

int main()
{
    // (x-2)^2 - x + 20sin(x) - 30cos(x)
#ifdef VECTOR
    vec_t val(1), grad(1);
    val << 15.8;
    std::function<double(const mf_t &)> func1 = [](const mf_t &val)->double{
        return arma::accu(arma::pow(val(0, 0) - 2, 2) - val(0, 0) + 20 * arma::sin(val(0, 0)) - 30 * arma::cos(val(0, 0)));
    };
    std::function<void(mf_t *, const mf_t &)> func2 = [](mf_t *grad, const mf_t &val)->void{
        (*grad)(0, 0) = 2 * val(0, 0) - 5 + 20 * arma::cos(val(0, 0)) + 30 * arma::sin(val(0, 0));
    };
#else
    mat_t val(2, 1), grad(2, 1);
    val = {0.0, 0.0};
    std::function<double(const mat_t &)> func1 = [](const mat_t &val)->double{
        return 100 * arma::sin(val(0, 0)) + 200 * arma::sin(val(1, 0));
    };
    std::function<void(mat_t *, const mat_t &)> func2 = [](mat_t *grad, const mat_t &val)->void{
        grad(0, 0) = 100 * arma::cos(val(0, 0));
        grad(1, 0) = 200 * arma::cos(val(1, 0));
    };
#endif

    mf_t mf_val(1, 1), mf_grad(1, 1);
    mf_val(0, 0) = val;
    mf_grad(0, 0) = grad;
    double obj = func1(mf_val);
    func2(&mf_grad, mf_val);

    optimization::l_bfgs::init_arg l_bfgs_args;
    l_bfgs_args.max_epoch = 50;
    unique_ptr<optimization::l_bfgs::optimizer<mf_t>> optm = make_unique<optimization::l_bfgs::optimizer<mf_t>>(l_bfgs_args, mf_val);

    cout << "val before: " << mf_val(0, 0);
    cout << "obj: " << obj << endl;
    cout << "grad: " << mf_grad(0, 0) << endl;
    optm->optimize(&mf_val, func1, func2);
    cout << "epoch past: " << optm->get_epoch() << endl;
    obj = func1(mf_val);
    func2(&mf_grad, mf_val);
    cout << "new_val: " << mf_val(0, 0);
    cout << "new_obj: " << obj << endl;
    cout << "new_grad: " << mf_grad(0, 0) << endl;

    return 0;
}

#endif

