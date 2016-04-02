#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cfloat>
#include <random>

using namespace std;

const string file1_data{"/Users/evensong/ml_data/regression/function1_sample"};
const string file1_test{"/Users/evensong/ml_data/regression/function1_sample_test"};
const string sinc_data{"/Users/evensong/ml_data/regression/sinc_sample"};
const string sinc_test{"/Users/evensong/ml_data/regression/sinc_sample_test"};

/*
 * y = sin(x) / x   (x != 0)
 * y = 1            (x == 1)
 */
void sinc() 
{
    const int NUM_SAMPLE = 5000;
    const int NUM_TEST_SAMPLE = 5000;
    fstream fs{sinc_data, ofstream::out};
    fs << NUM_SAMPLE << endl;

    std::uniform_real_distribution<double> dis_noise(-0.2, 0.2);
    std::uniform_real_distribution<double> dis_input(-10, 10);
    std::default_random_engine random;
    double y = 0.0;
    for (int i = 0; i < NUM_SAMPLE; ++i)
    {
        double error = dis_noise(random);
        double x = dis_input(random);
        if (std::abs(x - 0.0) < DBL_EPSILON)
            y = 1;
        else
            y = std::sin(x) / x + error;
        fs << x << ',' << y << endl;
    }
    fs.close();

    fs.open(sinc_test, ofstream::out);
    fs << NUM_TEST_SAMPLE << endl;

    for (int i = 0; i < NUM_TEST_SAMPLE; ++i)
    {
        double x = dis_input(random);
        if (std::abs(x - 0.0) < DBL_EPSILON)
            y = 1;
        else
            y = std::sin(x) / x;
        fs << x << ',' << y << endl;
    }
    fs.close();
}


/*
 * y = 0.7(sin(x))^3 + 0.3(y_old)^2 + e
 */
void experiment1() 
{
    const int NUM_SAMPLE = 1000;
    const int NUM_TEST_SAMPLE = 100;
    fstream fs{file1_data, ofstream::out};
    fs << NUM_SAMPLE << endl;

    std::uniform_real_distribution<double> dis_noise(-0.1, 0.1);
    std::uniform_real_distribution<double> dis_input(1, 11);
    std::default_random_engine random;
    double y_old = 0.0;
    double y = 0.0;
    for (int i = 0; i < NUM_SAMPLE; y_old = y, ++i)
    {
        double error = dis_noise(random);
        double x = dis_input(random);
        y = 0.7 * pow(sin(x), 3) + 0.3 * pow(y_old, 2) + error;
        fs << x << ',' << y << endl;
    }
    fs.close();

    fs.open(file1_test, ofstream::out);
    fs << NUM_TEST_SAMPLE << endl;

    y_old = 0.0;
    for (int i = 0; i < NUM_TEST_SAMPLE; y_old = y, ++i)
    {
//        double error = dis_noise(random);
        double x = dis_input(random);
        //y = 0.7 * pow(sin(x), 3) + 0.3 * pow(y_old, 2) + error;
        y = 0.7 * pow(sin(x), 3) + 0.3 * pow(y_old, 2);
        fs << x << ',' << y << endl;
    }
    fs.close();
}

int main()
{
    //experiment1();
    sinc();
    return 0;
}
