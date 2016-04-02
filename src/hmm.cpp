#include "../include/hmm.h"

int main()
{
    const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/nlp/icwb2-data/training/pku_training_encoded.utf8"};
    nlp::icwb2::read_data(TRAINING_DATA_PATH);

    return 0;

}
