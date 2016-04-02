#include "../include/stdafx.h"
#include "../include/prepare_data.h"
#include <clocale>
#include <codecvt>

//#define DEBUG

namespace nlp
{

namespace icwb2
{

unique_ptr<vector<string>> read_data(const string &path)
{
    //typedef std::codecvt_utf8<wchar_t> converter_type;
    //const unique_ptr<converter_type> converter = make_unique<converter_type>();
    //const std::locale utf8_locale = std::locale("chs");
    std::wifstream fs(path);
    //fs.imbue(utf8_locale);
    std::wstring each_line;
    
    //fs.open(path);
    //if (!fs.is_open())
    //    throw runtime_error("training examples file is wrong");

    getline(fs, each_line);
    std::wcout << each_line << endl;
    wchar_t c1 = each_line[0];
    for (int i = 0; i < each_line.length(); ++i)
        cout << i << endl;

    return nullptr;
}

} // namespace icwb2

} // namespace nlp


namespace artificial_data
{

unique_ptr<pair<mat_t, vec_t>> read_data(const string &path)
{
    string each_line;
    std::ifstream fs;
    int num_data;
    int num_attr;

    fs.open(path);
    if (!fs.is_open())
        throw runtime_error("training examples file is wrong");

    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &num_data);
    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &num_attr);
    unique_ptr<pair<mat_t, vec_t>> dataset = make_unique<pair<mat_t, vec_t>>(
            make_pair(mat_t(num_attr, num_data), vec_t(num_data)));

    int idx_data = 0;
    while (getline(fs, each_line))
    {
        int idx_attr = 0;
        auto it_begin = each_line.begin();
        auto it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
    
        double tmp;
        for (; idx_attr < num_attr; ++idx_attr)
        {
            std::sscanf(string{it_begin, it_end}.c_str(), "%lf", &tmp);
            dataset->first(idx_attr, idx_data) = tmp;
            it_begin = it_end;
            ++it_begin;
            it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
        }
        std::sscanf(string{it_begin, it_end}.c_str(), "%lf", &tmp);
        dataset->second(idx_data) = tmp;
        ++idx_data;
    }
    fs.close();

    return dataset;

}

} // namespace artificial_data

namespace uci
{

unique_ptr<attr_info> read_attr(const string &path)
{
    string each_line;
    std::ifstream fs;

    fs.open(path);
    if (!fs.is_open())
        CRY("attributes file wrong");

    int num_num_attr;
    int num_ctgr_attr;
    int dim_output;
    bool is_num_result = false;
    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &num_num_attr);
    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &num_ctgr_attr);
    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &dim_output);

    getline(fs, each_line);
    if (each_line == TAG_NUM_RESULT)
        is_num_result = true;
    else if (each_line != TAG_CTGR_RESULT)
        CRY("unknown result type");

    unique_ptr<attr_info> attribute = make_unique<attr_info>(num_num_attr, num_ctgr_attr, dim_output, is_num_result);
    while (getline(fs, each_line))
    {
        vector<string> line;
        line.reserve(32);
        auto it_begin = each_line.begin();
        auto it_end = std::find(it_begin, each_line.end(), DLM_ATTR_NAME);
        if (it_end == each_line.end())
        {
            if (is_num_result)
                break;
            it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
            while (it_end != each_line.end())
            {
                attribute->ctgr_result.emplace_back(it_begin, it_end);
                it_begin = it_end;
                ++it_begin;
                it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
            }
            attribute->ctgr_result.emplace_back(it_begin, it_end);
            break;
        }

        string name{it_begin, it_end};
        it_begin = it_end;
        while (it_begin != each_line.end())
        {
            ++it_begin;
            it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
            line.emplace_back(it_begin, it_end);
            it_begin = it_end;
        }
        if (line[0] == TAG_NUM_ATTR)
            attribute->num_attr_name.push_back(name);
        else
        {
            attribute->ctgr_attr_name.push_back(name);
            attribute->ctgr_attr_val.push_back(line);
        }
    }
    fs.close();

    return attribute;
}

unique_ptr<classification_data> read_data(const string &path, const attr_info &attribute)
{
    string each_line;
    std::ifstream fs;
    int num_data;
    const int num_num_attr = attribute.num_attr_name.size();
    const int num_ctgr_attr = attribute.ctgr_attr_name.size();

    fs.open(path);
    if (!fs.is_open())
        throw runtime_error("training examples file wrong");

    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &num_data);
    unique_ptr<classification_data> dataset = make_unique<classification_data>(attribute, num_data);

    int idx_data = 0;
    while (getline(fs, each_line))
    {
        int idx_attr = 0;
        auto it_begin = each_line.begin();
        auto it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
    
        for (; idx_attr < num_num_attr; ++idx_attr)
        {
            double tmp;
            std::sscanf(string{it_begin, it_end}.c_str(), "%lf", &tmp);
            dataset->input(idx_attr, idx_data) = tmp;
            it_begin = it_end;
            ++it_begin;
            it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
        }

        string ctgr_val;
        while (it_end != each_line.end())
        {
            ctgr_val.assign(it_begin, it_end);
            int pos = ctgr2pos(attribute, ctgr_val, idx_attr - num_num_attr);
            dataset->input(idx_attr, idx_data) = pos;
            it_begin = it_end;
            ++it_begin;
            it_end = std::find(it_begin, each_line.end(), DLM_ATTR_VAL);
            ++idx_attr;
        }
 
        if (attribute.is_num_result)
        {
            int tmp;
            std::sscanf(string{it_begin, it_end}.c_str(), "%d", &tmp);
            dataset->output.push_back(tmp);
        }
        else
        {
            string tmp{it_begin, it_end};
            int pos = result2pos(attribute, tmp);
            dataset->output.push_back(pos);
        }
        ++idx_data;
    }
    fs.close();

    return dataset;
}

void normalize(classification_data *dataset)
{
    for (int i = 0; i < dataset->input.n_rows; ++i)
    {
        const double max_val = dataset->input.row(i).max();
        const double min_val = dataset->input.row(i).min();
        if (std::abs(max_val - min_val) < DBL_EPSILON)
            if (std::abs(max_val - 0.0) < DBL_EPSILON)
                continue;
            else
                dataset->input.row(i) /= max_val;
        else
        {
            if (min_val < 0.0)
                dataset->input.row(i) += min_val;
            dataset->input.row(i) /= (max_val - min_val) / 2;
            dataset->input.row(i) -= 1.0;
        }
    }
}

pair<unique_ptr<vector<classification_data>>, unique_ptr<vector<classification_data>>> init_cross_validation(
        const attr_info &attribute, const classification_data &raw_data, int num_fold)
{
    if (num_fold <= 0)
        CRY();
    const int num_total_data = raw_data.input.n_cols;
    if (num_fold > num_total_data)
        num_fold = num_total_data;

    int num_testing_data;
    if (num_fold > 1)
        num_testing_data = num_total_data / num_fold;
    else
        num_testing_data = std::ceil(num_total_data * 0.2);
    int num_training_data = num_total_data - num_testing_data;
    unique_ptr<vector<classification_data>> training_data = make_unique<vector<classification_data>>(num_fold, classification_data(attribute, num_training_data));
    unique_ptr<vector<classification_data>> testing_data = make_unique<vector<classification_data>>(num_fold, classification_data(attribute, num_testing_data));
    unique_ptr<vector<int>> idx_data = make_unique<vector<int>>(num_total_data);

    for (int i = 0; i < num_total_data; ++i)
        (*idx_data)[i] = i;
    std::random_shuffle(idx_data->begin(), idx_data->end());
    int idx_testing_begin = 0;
    int idx_testing_end = num_testing_data;

    for (int idx_fold = 0; idx_fold < num_fold; ++idx_fold)
    {
        for (int i = 0; i < idx_testing_begin; ++i)
        {
            const int j = (*idx_data)[i];
            (*training_data)[idx_fold].input.col(i) = raw_data.input.col(j);
            (*training_data)[idx_fold].output.push_back(raw_data.output[j]);
        }
        
        for (int i = idx_testing_begin; i < idx_testing_end; ++i)
        {
            const int j = (*idx_data)[i];
            (*testing_data)[idx_fold].input.col(i - idx_testing_begin) = raw_data.input.col((*idx_data)[j]);
            (*testing_data)[idx_fold].output.push_back(raw_data.output[j]);
        }

        for (int i = idx_testing_end; i < num_total_data; ++i)
        {
            const int j = (*idx_data)[i];
            (*training_data)[idx_fold].input.col(i - num_testing_data) = raw_data.input.col((*idx_data)[j]);
            (*training_data)[idx_fold].output.push_back(raw_data.output[j]);
        }
        idx_testing_begin = idx_testing_end;
        idx_testing_end += num_testing_data;
    }

    return std::make_pair(std::move(training_data), std::move(testing_data));
}

} // namespace uci

namespace mnist
{

unique_ptr<cube_t> read_image(const string &path, const set_t set_type, const bool flip_gray_scale)
{
    std::ifstream fs(path, std::ios_base::in | std::ios_base::binary);
    if (!fs.is_open())
        throw std::runtime_error("image file " + path + "cannot open");

    unsigned char file_header[4];
    fs.read(reinterpret_cast<char *>(file_header), SIZE_FILE_HEADER);

    if (IMAGE_MAGIC_NUM != mnist::char2int(file_header))
        throw std::runtime_error(path + "is not a image file");

    // jump the useless information
    fs.read(reinterpret_cast<char *>(file_header), 4);
    fs.read(reinterpret_cast<char *>(file_header), 4);
    fs.read(reinterpret_cast<char *>(file_header), 4);

    const int image_size = IMAGE_ROWSIZE * IMAGE_COLSIZE;
    int idx_start;
    int total_image;
    switch (set_type)
    {
        case TRAINING:
            idx_start = 0;
            total_image = NUM_TRAINING_IMAGE;
            break;
        case VALIDATION:
            idx_start = NUM_TRAINING_IMAGE;
            total_image = NUM_VALIDATION_IMAGE;
            break;
        case TESTING:
            idx_start = 0;
            total_image = NUM_TESTING_IMAGE;
            break;
        default:
            throw runtime_error("in read_image(), wrong set_type\n");
            break;
    }
    unique_ptr<cube_t> image = make_unique<cube_t>(IMAGE_ROWSIZE, IMAGE_COLSIZE, total_image, arma::fill::zeros);
    fs.seekg(idx_start * image_size, fs.cur);
    for (int n = 0; n < total_image; ++n)
        for (int i = 0; i < IMAGE_ROWSIZE; ++i)
            for (int j = 0; j < IMAGE_COLSIZE; ++j)
            {
                unsigned char tmp;
                fs.read(reinterpret_cast<char *>(&tmp), 1);
                if (flip_gray_scale)
                    tmp = 255 - tmp;
                (*image)(i, j, n) = static_cast<double>(tmp);
            }

    return image;
}

unique_ptr<cube_t> read_blank_edge_image(const string &path, const int blank_size, const set_t set_type, const bool flip_gray_scale)
{
    if (blank_size <= 0)
        throw std::runtime_error("negative or zero blank size");

    std::ifstream fs(path, std::ios_base::in | std::ios_base::binary);
    if (!fs.is_open())
        throw std::runtime_error("image file " + path + "cannot open");

    unsigned char file_header[4];
    fs.read(reinterpret_cast<char *>(file_header), SIZE_FILE_HEADER);

    if (IMAGE_MAGIC_NUM != mnist::char2int(file_header))
        throw std::runtime_error(path + "is not a image file");

    // jump the useless information
    fs.read(reinterpret_cast<char *>(file_header), 4);
    fs.read(reinterpret_cast<char *>(file_header), 4);
    fs.read(reinterpret_cast<char *>(file_header), 4);

    const int raw_image_size = IMAGE_ROWSIZE * IMAGE_COLSIZE;
    const int image_row = IMAGE_ROWSIZE + 2 * blank_size;
    const int image_col = IMAGE_COLSIZE + 2 * blank_size;
    int idx_start;
    int total_image;
    switch (set_type)
    {
        case TRAINING:
            idx_start = 0;
            total_image = NUM_TRAINING_IMAGE;
            break;
        case VALIDATION:
            idx_start = NUM_TRAINING_IMAGE;
            total_image = NUM_VALIDATION_IMAGE;
            break;
        case TESTING:
            idx_start = 0;
            total_image = NUM_TESTING_IMAGE;
            break;
        default:
            throw runtime_error("in read_image(), wrong set_type\n");
            break;
    }
    unique_ptr<cube_t> image;
    if (flip_gray_scale)
    {
        image = make_unique<cube_t>(image_row, image_col, total_image);
        image->fill(255.0);
    }
    else
        image = make_unique<cube_t>(image_row, image_col, total_image, arma::fill::zeros);
    fs.seekg(idx_start * raw_image_size, fs.cur);
    for (int n = 0; n < total_image; ++n)
        for (int i = blank_size; i < blank_size + IMAGE_ROWSIZE; ++i)
            for (int j = blank_size; j < blank_size + IMAGE_COLSIZE; ++j)
            {
                unsigned char tmp;
                fs.read(reinterpret_cast<char *>(&tmp), 1);
                if (flip_gray_scale)
                    tmp = 255 - tmp;
                (*image)(i, j, n) = static_cast<double>(tmp);
            }
    return image;
}

unique_ptr<mat_t> read_vectorized_image(const string &path, const set_t set_type, const bool flip_gray_scale)
{
    std::ifstream fs(path, std::ios_base::in | std::ios_base::binary);
    if (!fs.is_open())
        throw std::runtime_error("image file " + path + "cannot open");

    unsigned char file_header[4];
    fs.read(reinterpret_cast<char *>(file_header), SIZE_FILE_HEADER);

    if (IMAGE_MAGIC_NUM != mnist::char2int(file_header))
        throw std::runtime_error(path + "is not a image file");

    // jump the useless information
    fs.read(reinterpret_cast<char *>(file_header), 4);
    fs.read(reinterpret_cast<char *>(file_header), 4);
    fs.read(reinterpret_cast<char *>(file_header), 4);

    const int image_size = IMAGE_ROWSIZE * IMAGE_COLSIZE;
    int idx_start;
    int total_image;
    switch (set_type)
    {
        case TRAINING:
            idx_start = 0;
            total_image = NUM_TRAINING_IMAGE;
            break;
        case VALIDATION:
            idx_start = NUM_TRAINING_IMAGE;
            total_image = NUM_VALIDATION_IMAGE;
            break;
        case TESTING:
            idx_start = 0;
            total_image = NUM_TESTING_IMAGE;
            break;
        default:
            throw runtime_error("in read_image(), wrong set_type\n");
            break;
    }
    unique_ptr<mat_t> image = make_unique<mat_t>(image_size, total_image, arma::fill::zeros);
    fs.seekg(idx_start * image_size, fs.cur);
    for (int i = 0; i < total_image; ++i)
    {
        std::generate_n(image->begin_col(i), image_size, [&fs, flip_gray_scale]()->double
        {
            unsigned char tmp;
            fs.read(reinterpret_cast<char *>(&tmp), 1);
            if (flip_gray_scale)
                tmp = 255 - tmp;
            return static_cast<double>(tmp);
        });
    }
    return image;
}

unique_ptr<vector<int>> read_label(const string &path, const set_t set_type)
{
    std::ifstream fs(path, std::ios_base::in | std::ios_base::binary);
    if (!fs.is_open())
        throw std::runtime_error("label file " + path + "cannot open");

    unsigned char file_header[4];
    fs.read(reinterpret_cast<char *>(file_header), SIZE_FILE_HEADER);
    if (LABEL_MAGIC_NUM != mnist::char2int(file_header))
        throw std::runtime_error(path + "is not a label file");

    fs.read(reinterpret_cast<char *>(file_header), 4); // jump the useless information
 
    int total_label;
    int idx_start;
    switch (set_type)
    {
        case TRAINING:
            idx_start = 0;
            total_label = NUM_TRAINING_LABEL;
            break;
        case VALIDATION:
            idx_start = NUM_TRAINING_LABEL;
            total_label = NUM_VALIDATION_LABEL;
            break;
        case TESTING:
            idx_start = 0;
            total_label = NUM_TESTING_LABEL;
            break;
        default:
            throw runtime_error("in read_label(), wrong set_type\n");
            break;
    }
    unique_ptr<vector<int>> label(new vector<int>(total_label));
    fs.seekg(idx_start, fs.cur);
    generate_n(label->begin(), total_label, [&fs]()->int
    {
        unsigned char tmp;
        fs.read(reinterpret_cast<char *>(&tmp), 1);
        return static_cast<int>(tmp);
    });
    return label;
}

} // namespace mnist

namespace image_processor
{

distorter::distorter(const int image_rows, const int image_cols, const double elastic_factor, const double scaling_factor, const double rot_factor) :
    random_seed(std::chrono::system_clock::now().time_since_epoch().count()),
    random_engine(random_seed)
{
    IMAGE_ROWS = image_rows;
    IMAGE_COLS = image_cols;
    ELASTIC_FACTOR = elastic_factor;
    SCALING_FACTOR = scaling_factor;
    ROT_FACTOR = rot_factor;
    GAUSSIAN_FIELD_SIZE = 21;
    ELASTIC_SIGMA = 8.0; // higher numbers mean less distortion, 4.0 may be okay

    GAUSSIAN_VAR = 1.0 / (2.0 * ELASTIC_SIGMA * ELASTIC_SIGMA);
    GAUSSIAN_COEFF = 1.0 / (std::sqrt(2.0 * 3.1415926535897932384626433832795) * ELASTIC_SIGMA);
    uniform_dist = std::uniform_real_distribution<double>(-1, 1);
    get_uni = [this]()->double { return uniform_dist(random_engine); }; 

    gaussian_ker = make_unique<mat_t>(GAUSSIAN_FIELD_SIZE, GAUSSIAN_FIELD_SIZE, arma::fill::zeros);
    row_uni_dist = make_unique<mat_t>(IMAGE_ROWS, IMAGE_COLS, arma::fill::zeros);
    col_uni_dist = make_unique<mat_t>(IMAGE_ROWS, IMAGE_COLS, arma::fill::zeros);

    row_map = make_unique<mat_t>(IMAGE_ROWS, IMAGE_COLS, arma::fill::zeros);
    col_map = make_unique<mat_t>(IMAGE_ROWS, IMAGE_COLS, arma::fill::zeros);
    mapped_img = make_unique<mat_t>(IMAGE_ROWS, IMAGE_COLS, arma::fill::zeros);

    for (int row = 0, half_width = GAUSSIAN_FIELD_SIZE / 2; row < gaussian_ker->n_rows; ++row)
        for (int col = 0; col < gaussian_ker->n_cols; ++col)
            (*gaussian_ker)(row, col) = GAUSSIAN_COEFF * std::exp(-((std::pow(row - half_width, 2) + std::pow(col - half_width, 2)) * GAUSSIAN_VAR));
}

void distorter::generate_map(const double severity_factor)
{
    fill_uni(row_uni_dist.get());
    fill_uni(col_uni_dist.get());

    for (int i = 0, half_width = GAUSSIAN_FIELD_SIZE / 2; i < IMAGE_ROWS; ++i) 
        for (int j = 0; j < IMAGE_COLS; ++j)
        {
            double r_result = 0.0, c_result = 0.0;
            int r_uni = 0, c_uni = 0;

            for (int r_gaussian = 0; r_gaussian < GAUSSIAN_FIELD_SIZE; ++r_gaussian)
                for (int c_gaussian = 0; c_gaussian < GAUSSIAN_FIELD_SIZE; ++c_gaussian)
                {
                    r_uni = i - half_width + r_gaussian;
                    c_uni = j - half_width + c_gaussian;
                    if (r_uni < 0 || r_uni >= IMAGE_ROWS || c_uni < 0 || c_uni >= IMAGE_COLS)
                        continue;
                    else
                    {
                        r_result += (*row_uni_dist)(r_uni, c_uni) * (*gaussian_ker)(r_gaussian, c_gaussian);
                        c_result += (*col_uni_dist)(r_uni, c_uni) * (*gaussian_ker)(r_gaussian, c_gaussian);
                    }
                }
            
            (*row_map)(i, j) = ELASTIC_FACTOR * r_result;
            (*col_map)(i, j) = ELASTIC_FACTOR * c_result;
        }

    const double row_scale_fac = severity_factor * SCALING_FACTOR / 100.0 * get_uni();
    const double col_scale_fac = severity_factor * SCALING_FACTOR / 100.0 * get_uni();
    for (int i = 0, half_width = IMAGE_ROWS / 2; i < IMAGE_ROWS; ++i) 
        for (int j = 0; j < IMAGE_COLS; ++j)
        {
            (*row_map)(i, j) += row_scale_fac * (j - half_width);
            (*col_map)(i, j) -= col_scale_fac * (half_width - i);
        }

    double angle = severity_factor * ROT_FACTOR * get_uni();
    angle *= 3.1415926535897932384626433832795 / 180.0;
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);

    for (int i = 0, half_width = IMAGE_ROWS / 2; i < IMAGE_ROWS; ++i) 
        for (int j = 0; j < IMAGE_COLS; ++j)
        {
            (*row_map)(i, j) += (j - half_width) * (cos_angle - 1) - (half_width - i) * sin_angle;
            (*col_map)(i, j) -= (half_width - i) * (cos_angle - 1) + (j - half_width) * sin_angle;
        }
}

void distorter::mapping(mat_t *image_src)
{
    // start bilinear interpolation
    mapped_img->zeros();
    double src_row, src_col;
    double frac_row, frac_col;
    double w1, w2, w3, w4;
    double src_val;
    int s_row, s_col, s_rowp1, s_colp1;
    bool is_out_of_bound;

    for (int i = 0; i < IMAGE_ROWS; ++i)
        for (int j = 0; j < IMAGE_COLS; ++j)
        {
            src_row = static_cast<double>(i) - (*col_map)(i, j);
            src_col = static_cast<double>(j) - (*row_map)(i, j);

            frac_row = src_row - static_cast<int>(src_row);
            frac_col = src_col - static_cast<int>(src_col);

            w1 = (1.0 - frac_row) * (1.0 - frac_col);
            w2 = (1.0 - frac_row) * frac_col;
            w3 = frac_row * (1.0 - frac_col);
            w4 = frac_row * frac_col;

            is_out_of_bound = false;
            if (src_row + 1.0 >= IMAGE_ROWS || src_row < 0 ||
                    src_col + 1.0 >= IMAGE_COLS || src_col < 0)
                is_out_of_bound = true;

            if (!is_out_of_bound)
            {
                s_row = static_cast<int>(src_row);
                s_col = static_cast<int>(src_col);
                s_rowp1 = s_row + 1;
                s_colp1 = s_col + 1;

                while (s_rowp1 >= IMAGE_ROWS)
                    s_rowp1 -= IMAGE_ROWS;
                while (s_rowp1 < 0)
                    s_rowp1 += IMAGE_ROWS;
                while (s_colp1 >= IMAGE_COLS)
                    s_colp1 -= IMAGE_COLS;
                while (s_colp1 < 0)
                    s_colp1 += IMAGE_COLS;

                src_val = w1 * (*image_src)(s_row, s_col) +
                    w2 * (*image_src)(s_row, s_colp1) +
                    w3 * (*image_src)(s_rowp1, s_col) +
                    w4 * (*image_src)(s_rowp1, s_colp1);
            }
            else
                src_val = 1.0;

            (*mapped_img)(i, j) = 0.5 * (1.0 - src_val);
        }

    for (int i = 0; i < IMAGE_ROWS; ++i)
        for (int j = 0; j < IMAGE_COLS; ++j)
            (*image_src)(i, j) = 1.0 - 2.0 * (*mapped_img)(i, j);
}

} // namespace image_processor


