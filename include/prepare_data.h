#ifndef EVENSONG_PREPARE_DATA
#define EVENSONG_PREPARE_DATA

#include "./stdafx.h"

namespace artificial_data
{
    
const char DLM_ATTR_VAL = ',';
unique_ptr<pair<mat_t, vec_t>> read_data(const string &path);

} // namespace function_data

namespace uci
{

using std::unordered_multimap;
using std::iterator;

const string TAG_NUM_RESULT{"NUM_RESULT"};
const string TAG_CTGR_RESULT{"CTGR_RESULT"};
const string TAG_NUM_ATTR{"continuous"};
const char DLM_ATTR_NAME = ':';
const char DLM_ATTR_VAL = ',';

struct attr_info
{
    attr_info(const int num_num_attr, const int num_ctgr_attr, const int _dim_output, bool result_type) :
        is_num_result(result_type),
        dim_input(num_num_attr + num_ctgr_attr),
        dim_output(_dim_output)
    {
        num_attr_name.reserve(num_num_attr);
        ctgr_attr_name.reserve(num_ctgr_attr);
        ctgr_attr_val.reserve(num_ctgr_attr);
        if (!is_num_result)
            ctgr_result.reserve(dim_output);
    }

    bool is_num_result;
    int dim_input;
    int dim_output;
    vector<string> num_attr_name;
    vector<string> ctgr_attr_name;
    vector<vector<string>> ctgr_attr_val;
    vector<string> ctgr_result;
};

//
// used for supervised learning
//
struct classification_data
{
    classification_data(const attr_info &attribute, const int num_data) :
        num_num_attr(attribute.num_attr_name.size()),
        num_ctgr_attr(attribute.ctgr_attr_name.size()),
        input(attribute.dim_input, num_data)
    {
        output.reserve(num_data);
    }

    int num_num_attr;
    int num_ctgr_attr;
    mat_t input;
    vector<int> output;
};

inline int ctgr2pos(const attr_info &attribute, const string &ctgr_val, const int idx_attr)
{
    int i = 0;
    for (const string &e : attribute.ctgr_attr_val[idx_attr])
    {
        if (e == ctgr_val)
            break;
        ++i;
    }

    return i;
}

inline int result2pos(const attr_info &attribute, const string &val)
{
    int i = 0;
    for (const string &e : attribute.ctgr_result)
    {
        if (e == val)
            break;
        ++i;
    }

    return i;
}

unique_ptr<attr_info> read_attr(const string &path);

unique_ptr<classification_data> read_data(const string &path, const attr_info &attribute);

void normalize(classification_data *dataset);

inline void normalize(vector<classification_data> *dataset)
{
    for (auto &e : *dataset)
        normalize(&e);
}

pair<unique_ptr<vector<classification_data>>, unique_ptr<vector<classification_data>>> init_cross_validation(
        const attr_info &attribute, const classification_data &raw_data, const int num_fold);

} // namespace uci

namespace mnist
{

const int IMAGE_MAGIC_NUM = 2051;
const int LABEL_MAGIC_NUM = 2049;

const int NUM_TRAINING_IMAGE = 50000;
const int NUM_TRAINING_LABEL = 50000;
const int NUM_VALIDATION_IMAGE = 10000;
const int NUM_VALIDATION_LABEL = 10000;
const int NUM_TESTING_IMAGE = 10000;
const int NUM_TESTING_LABEL = 10000;
/*
const int NUM_TRAINING_IMAGE = 60000;
const int NUM_TRAINING_LABEL = 60000;
const int NUM_TESTING_IMAGE = 10000;
const int NUM_TESTING_LABEL = 10000;
const int NUM_VALIDATION_IMAGE = 0;
const int NUM_VALIDATION_LABEL = 0;
*/
enum set_t { TRAINING = 1, VALIDATION = 2, TESTING = 3 };

const int NUM_LABEL_TYPE = 10;
const int IMAGE_ROWSIZE = 28;
const int IMAGE_COLSIZE = 28;
const int SIZE_FILE_HEADER = 4;

const double MIN_PIXEL_VALUE = 0.0;
const double MAX_PIXEL_VALUE = 255.0;
const double BINARY_THRESHOLD = 20.0;

// big-endian to small-endian
inline int char2int(const unsigned char *arr) 
{
    return ((static_cast<int>(arr[0]) << 24) + (static_cast<int>(arr[1]) << 16) + (static_cast<int>(arr[2]) << 8) + static_cast<int>(arr[3]));
}

unique_ptr<cube_t> read_image(const string &path, const set_t set_type, const bool flip_gray_scale = false);

unique_ptr<cube_t> read_blank_edge_image(const string &path, const int blank_size, const set_t set_type, const bool filp_gray_scale = false);

unique_ptr<mat_t> read_vectorized_image(const string &path, const set_t set_type, const bool flip_gray_scale = false);

unique_ptr<vector<int>> read_label(const string &path, const set_t set_type);

// pixels are normalized to -1 ~ 1
// -1 is black
// 1 is white
template <typename VAL_T>
inline void normalize(VAL_T *image_src)
{
    std::transform(image_src->begin(), image_src->end(), image_src->begin(), [](double e)->double
    {
        if (std::abs(e - 255.0) < DBL_EPSILON)
            e = 256.0;
        return e / 128.0 - 1.0;
    });
}

} // namespace mnist

namespace image_processor
{

class distorter
{
    public:
        // elastic_factor: higher numbers amplify the distortions; between 0 ~ 1 
        // scaling_factor: 15.0 for 15%
        // rot_factor: 15.0 for 15 degrees
        distorter(const int image_rows, const int image_col, const double elastic_factor = 0.5, const double scaling_factor = 15.0, const double rot_factor = 15.0);

        // severity_factor: > 0
        // 1.0 is the standard severity
        // if we only want half as harsh a distortion, set severity_factor to 0.5
        void process(mat_t *image_src, const double severity_factor = 1.0)
        {
            generate_map(severity_factor);
            mapping(image_src);
        }

    private:
        void generate_map(const double severity_factor);
        void mapping(mat_t *image_src);

        void fill_uni(mat_t *matrix)
        {
            matrix->imbue(get_uni);
        }

        std::function<double()> get_uni;
        int IMAGE_ROWS;
        int IMAGE_COLS;
        double ELASTIC_FACTOR;
        double SCALING_FACTOR;
        double ROT_FACTOR;
        int GAUSSIAN_FIELD_SIZE;
        double ELASTIC_SIGMA; // a higher number means less distortion, 8.0 is applied, and 4.0 may be okay

        double GAUSSIAN_VAR;
        double GAUSSIAN_COEFF;
        unsigned int random_seed;
        std::uniform_real_distribution<double> uniform_dist;
        std::default_random_engine random_engine;
        unique_ptr<mat_t> row_uni_dist;
        unique_ptr<mat_t> col_uni_dist;

        unique_ptr<mat_t> gaussian_ker;
        unique_ptr<mat_t> row_map;
        unique_ptr<mat_t> col_map;
        unique_ptr<mat_t> mapped_img;
};

} // namespace image_processor

namespace nlp
{

namespace icwb2
{

unique_ptr<vector<string>> read_data(const string &path);

} // namespace icwb2

} // namespace nlp

#endif


