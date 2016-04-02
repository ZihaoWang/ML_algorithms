#include "../include/tree.h"

#define UNIT_TESTING

namespace c4_5
{

double tree::test_tree()
{
    cout << "----------------------------start testing" << endl;
    int num_correct = 0;

    for (int i = 0; i < NUM_TESTING_DATA; ++i)
    {
        weak_ptr<node> node_now(decision_tree);
        bool is_missing_val = false;
        while (node_now.lock()->node_type != TAG_LEAF_NODE)
        {
            string val_now = test_exms->attr[i][node_now.lock()->idx_attr_name];
            if (val_now == MISSING_VALUE) // if any testing example has missing attribute values, we just consider this example would be classified wrongly.
            {
                is_missing_val = true;
                break;
            }
            node_now = node_now.lock()->get_proper_child(val_now);
            if (node_now.lock().get() == nullptr) // if an attribute value didn't occur in training examples, the corresponding pointer is nullptr. 
                                            // if any testing example has such attribute value, we just consider this example would be classified wrongly.
                break;
        }
        
        if (!is_missing_val && node_now.lock().get() != nullptr)
        {
            weak_ptr<leaf> leaf_node = static_pointer_cast<leaf>(node_now.lock());
            if (test_exms->result[i] == leaf_node.lock()->attr_val)
                ++num_correct;
        }
    }
    return static_cast<double>(num_correct) / NUM_TESTING_DATA;
}

tuple<string, int, bool> tree::comp_best_attr()
{
    const int POSITIVE_SUM = count(exms->result.begin(), exms->result.end(), POSITIVE_VAL);
    const int NUM_EXAMPLE = exms->result.size();
    const int NUM_ATTR_NAME = attributes->size();
    const double prb_pst_all = static_cast<double>(POSITIVE_SUM) / exms->result.size();
    const double prb_neg_all = 1.0 - prb_pst_all;

    if (abs(prb_pst_all) < DBL_EPSILON || abs(prb_neg_all) < DBL_EPSILON)
        return make_tuple("", 0, false);
    const double current_entropy = -1 * (prb_pst_all * log2(prb_pst_all) + prb_neg_all * log2(prb_neg_all));

    vector<double> gains;
    gains.reserve(attributes->size());
    for (int i = 0; i < attributes->size(); ++i)
        gains.push_back(0.0);
    vector<string> delimiter;
    delimiter.reserve(attributes->size());
    for (int i = 0; i < attributes->size(); ++i)
        delimiter.push_back("");

    for (int idx_attr_name = 0; idx_attr_name < NUM_ATTR_NAME; ++idx_attr_name)
    {
        vector<double> each_val_entropy; 
        vector<int> num_each_val;
        double attr_entropy = 0.0; // c4.5 algorithm 
         
        if (is_active(idx_attr_name))
        {
            int attr_val_num = (*attributes)[idx_attr_name].size() - 1;
            if ((*attributes)[idx_attr_name][1] == TAG_NUM_ATTR)
                attr_val_num = 1; // numeric attribute only have 2 children (1 delimiter)

            each_val_entropy.reserve(attr_val_num);
            num_each_val.reserve(attr_val_num + 1);
            for (int i = 0; i < attr_val_num; ++i)
            {
                each_val_entropy.push_back(0.0);
                num_each_val.push_back(0);
            }
            num_each_val.push_back(0);

            for (int idx_attr_val = 1; idx_attr_val <= attr_val_num; ++idx_attr_val)
            {
                string attr_val = (*attributes)[idx_attr_name][idx_attr_val];

                if (attr_val == TAG_NUM_ATTR) // for numeric attribute
                {
                    auto vals = make_unique<vector<pair<double, string>>>();
                    for (int i = 0; i < NUM_EXAMPLE; ++i)
                    {
                        double tmp;
                        sscanf(exms->attr[i][idx_attr_name].c_str(), "%lf", &tmp);
                        vals->push_back(make_pair(tmp, exms->result[i]));
                    }

                    sort(vals->begin(), vals->end(), [](const pair<double, string> a, const pair<double, string> b)->bool{
                            return a.first < b.first;
                            });

                    int num_ls_pst = 0;
                    int idx_exm = 0;
                    for (auto iter_now = vals->begin(); iter_now != vals->end() - 1; ++iter_now, ++idx_exm)
                    {
                        if (abs(iter_now->first - (iter_now + 1)->first) < DBL_EPSILON) // ignore the same balue
                        {
                            if (iter_now->second == POSITIVE_VAL)
                                ++num_ls_pst;
                            continue;
                        }
                        if (iter_now->second == (iter_now + 1)->second) // find delimiters which have different results on both sides(< and >)
                        {
                            if (iter_now->second == POSITIVE_VAL)
                                ++num_ls_pst;
                            continue;
                        }
                        if (iter_now->second == POSITIVE_VAL)
                            ++num_ls_pst;

                        double prb_ls = (0.0 + idx_exm) / NUM_EXAMPLE; // the amount of examples less than delimiter
                        double prb_ls_pst = (idx_exm == 0) ? 0 : ((0.0 + num_ls_pst) / idx_exm); // the probability of positive results less than delimiter

                        double prb_gt_pst = (POSITIVE_SUM - num_ls_pst + 0.0) / (NUM_EXAMPLE - idx_exm); // ~ greater than delimiter
                        double entropy_ls = 0.0;
                        double entropy_gt = 0.0;
                        if ((abs(prb_ls_pst - 0.0) < DBL_EPSILON) || (abs(prb_ls_pst - 1.0) < DBL_EPSILON))
                            entropy_ls = 0.0;
                        else
                            entropy_ls = (-1) * (prb_ls_pst * log2(prb_ls_pst) + (1 - prb_ls_pst) * log2(1 - prb_ls_pst)) * prb_ls;
                        if ((abs(prb_gt_pst - 0.0) < DBL_EPSILON) || (abs(prb_gt_pst - 1.0) < DBL_EPSILON))
                            entropy_gt = 0.0;
                        else
                            entropy_gt = (-1) * (prb_gt_pst * log2(prb_gt_pst) + (1 - prb_gt_pst) * log2(1 - prb_gt_pst)) * (1 - prb_ls);
                        if (entropy_ls + entropy_gt < each_val_entropy[0] || (abs(each_val_entropy[0] - 0.0) < DBL_EPSILON))
                        {
                            each_val_entropy[0] = entropy_ls + entropy_gt;
                            num_each_val[0] = idx_exm;
                            num_each_val[1] = NUM_EXAMPLE - idx_exm;

                            char str_tmp[32];
                            sprintf(str_tmp, "%lf", iter_now->first);
                            delimiter[idx_attr_name] = string{str_tmp};
                            if ((abs(prb_ls - 0.0) < DBL_EPSILON) || (abs(prb_ls - 1.0) < DBL_EPSILON))
                                attr_entropy = 0.0;
                            else
                                attr_entropy = (-1) * (prb_ls * log2(prb_ls) + (1 - prb_ls) * log2(1 - prb_ls));
                        }
                    }
                }

                else // for categorical attribute
                {
                    int num_eq = 0; // total amount of examples which have the same value
                    int num_eq_pst = 0; // total amount of examples which have the same value and positive result
                    for (int idx_exm = 0; idx_exm < NUM_EXAMPLE; ++idx_exm)
                    {
                        if (exms->attr[idx_exm][idx_attr_name] == attr_val)
                        {
                            ++num_eq;
                            if (exms->result[idx_exm] == POSITIVE_VAL)
                                ++num_eq_pst;
                        }
                    }
                    num_each_val[idx_attr_val - 1] = num_eq;
                    double prb_eq_pst = 0.0;
                    if (num_eq != 0)
                        prb_eq_pst = (num_eq_pst + 0.0) / num_eq;
                    double prb_eq = (num_eq + 0.0) / NUM_EXAMPLE;
                    if (abs(prb_eq_pst - 0.0) < DBL_EPSILON || abs(prb_eq_pst - 1.0) < DBL_EPSILON)
                        each_val_entropy[idx_attr_val - 1] = 0;
                    else
                        each_val_entropy[idx_attr_val - 1] = (-1) * (prb_eq_pst * log2(prb_eq_pst) + (1 - prb_eq_pst) * log2(1 - prb_eq_pst)) * prb_eq;
                    if (abs(prb_eq - 0.0) < DBL_EPSILON)
                        ;
                    else
                        attr_entropy += (-1) * (prb_eq * log2(prb_eq));
                }
            }
            // C4.5 algorithm
            gains[idx_attr_name] = current_entropy - accumulate(each_val_entropy.begin(), each_val_entropy.end(), 0.0); 
            if (attr_entropy != 0)
                gains[idx_attr_name] /= attr_entropy;
            else
                gains[idx_attr_name] = 0.0;
        }
    }
    
    bool is_single_node;
    auto iter = max_element(gains.begin(), gains.end());
    if (*iter < SINGLE_NODE_THRESHOLD)
        is_single_node = true;
    else
        is_single_node = false;

    int idx_best_attr = iter - gains.begin();
    if ((*attributes)[idx_best_attr][1] == TAG_NUM_ATTR)
    {
        (*attributes)[idx_best_attr].push_back(delimiter[idx_best_attr]);
    }
        
    string attr_name{(*attributes)[idx_best_attr][0]};

    return make_tuple(attr_name, idx_best_attr, is_single_node);
}

shared_ptr<node> tree::create_node(shared_ptr<node> parent, const vector<int> &idx_exm, bool is_single_node, const int level)
{
    if (idx_exm.empty())
    {
        active_attrs->push_back(-1);
        return nullptr;
    }

    auto iter_begin = idx_exm.begin();
    auto iter = find_if(idx_exm.begin() + 1, idx_exm.end(), [&iter_begin, this](int e)->bool{
            return (exms->result[e] != exms->result[*iter_begin]) ? true : false;
            });
    if (iter == idx_exm.end()) // if all the examples have same result
    {
        auto new_node = make_shared<leaf>(parent);

        new_node->level = level;
        new_node->node_type = TAG_LEAF_NODE;
        new_node->attr_name = string{"leaf"};
        new_node->attr_val = exms->result[*iter_begin];
        new_node->pos_train_num = idx_exm.size();
        active_attrs->push_back(-1);
        return new_node;
    }

    iter = active_attrs->begin();
    if (!active_attrs->empty())
        iter = find(active_attrs->begin(), active_attrs->end(), -1);

    if (iter - active_attrs->begin() == attributes->size()/*1*/ || is_single_node) // if all the attributes are inactive or parent is a single node
    {
        auto new_node = make_shared<leaf>(parent);

        new_node->level = level;
        new_node->node_type = TAG_LEAF_NODE;
        new_node->attr_name = string{"leaf"};
        for (auto i : idx_exm)
            if (exms->result[i] == POSITIVE_VAL)
                ++new_node->pos_train_num;
            else
                ++new_node->neg_train_num;
        if (new_node->pos_train_num >= new_node->neg_train_num)
            new_node->attr_val = POSITIVE_VAL;
        else
            new_node->attr_val = NEGATIVE_VAL;

        active_attrs->push_back(-1);
        return new_node;
    }
    
    string attr_name;
    int idx_attr_name;
    tie(attr_name, idx_attr_name, is_single_node) = comp_best_attr();

    if ((*attributes)[idx_attr_name][1] == TAG_NUM_ATTR) // for numeric attributes
    {
        auto idx_l_exm = make_unique<vector<int>>();
        auto idx_r_exm = make_unique<vector<int>>();
        double attr_val;
        sscanf((*attributes)[idx_attr_name][2].c_str(), "%lf", &attr_val);

        auto new_node = make_shared<numeric_node>(parent);

        new_node->level = level;
        new_node->attr_name = attr_name;
        new_node->idx_attr_name = idx_attr_name;
        new_node->attr_val = attr_val;
        new_node->node_type = TAG_NUM_NODE;

        for (int i : idx_exm)
        {
            double each_val;
            sscanf((*exms).attr[i][idx_attr_name].c_str(), "%lf", &each_val);
            if (each_val > attr_val)
                idx_r_exm->push_back(i);
            else
                idx_l_exm->push_back(i);
        }
        
        active_attrs->push_back(idx_attr_name);
        new_node->lchild = create_node(new_node, *idx_l_exm, is_single_node, level + 1);
        active_attrs->pop_back();
        new_node->rchild = create_node(new_node, *idx_r_exm, is_single_node, level + 1);
        active_attrs->pop_back();

        return new_node;
    }

    else // for categorical attributes
    {
        auto new_node = make_shared<categorical_node>(parent);

        new_node->level = level;
        new_node->attr_name = attr_name;
        new_node->idx_attr_name = idx_attr_name;
        new_node->node_type = TAG_CTGR_NODE;

        if (is_single_node)
        {
            vector<int> num_attr_val((*attributes)[idx_attr_name].size() - 1, 0);

            // find out which value of this attribute appears most frequently
            for (auto e : idx_exm)
            {
                auto each_idx = find((*attributes)[idx_attr_name].begin(), (*attributes)[idx_attr_name].end(), exms->attr[e][idx_attr_name]);
                if (each_idx == (*attributes)[idx_attr_name].end()) // if unknown attribute value appears, we just throw this example away
                    continue;
                int idx = each_idx - (*attributes)[idx_attr_name].begin() - 1;
                ++num_attr_val[idx];
            }
            auto freq_idx = max_element(num_attr_val.begin(), num_attr_val.end()); 
            string attr_val = (*attributes)[idx_attr_name][freq_idx - num_attr_val.begin() + 1];
            
            auto idx_l_exm = make_unique<vector<int>>();
            auto idx_r_exm = make_unique<vector<int>>();
            for (auto e : idx_exm)
                if (exms->attr[e][idx_attr_name] == attr_val)
                    idx_l_exm->push_back(e);
                else
                    idx_r_exm->push_back(e);

            active_attrs->push_back(idx_attr_name);
            new_node->children[string{"?"}] = create_node(new_node, *idx_r_exm, true, level + 1);
            active_attrs->pop_back();
            new_node->children[attr_val] = create_node(new_node, *idx_l_exm, true, level + 1);
            active_attrs->pop_back();
        }

        else // for categorical attributes and not single node
        {
            auto idx_children = make_unique<vector<vector<int>>>();    
            int num_attr = (*attributes)[idx_attr_name].size() - 1;
            idx_children->reserve(num_attr);
            for (int i = 0; i < num_attr; ++i)
                idx_children->emplace_back();

            for (auto e : idx_exm)
            {
                auto each_idx = find((*attributes)[idx_attr_name].begin(), (*attributes)[idx_attr_name].end(), exms->attr[e][idx_attr_name]);
                if (each_idx == (*attributes)[idx_attr_name].end()) // if unknown attribute value appears, we just throw this example away
                    continue;
                int idx = each_idx - (*attributes)[idx_attr_name].begin() - 1;
                (*idx_children)[idx].push_back(e);
            }
            
            active_attrs->push_back(idx_attr_name);
            for (int i = 0; i < num_attr; ++i)
            {
                string attr_val = (*attributes)[idx_attr_name][i + 1];
                new_node->children[attr_val] = create_node(new_node, (*idx_children)[i], false, level + 1);
                active_attrs->pop_back();
            }
        } // if (is_single_node)
        return new_node;
    } // for categorical attributes
}

void tree::print_node(shared_ptr<node> tree)
{
    if (tree == nullptr)
    {
        return;
    }

    if (tree->node_type == TAG_NUM_NODE)
    {
        shared_ptr<numeric_node> tree_now = static_pointer_cast<numeric_node>(tree);

        print_node(tree_now->rchild);
        cout << *tree_now << endl;
        print_node(tree_now->lchild);
    }
    else if (tree->node_type == TAG_CTGR_NODE)
    {
        shared_ptr<categorical_node> tree_now = static_pointer_cast<categorical_node>(tree);
        int i = 0;

        auto citer = tree_now->children.rbegin();
        for (; i < tree_now->children.size() / 2; ++i, ++citer)
        {
            if (citer->second == nullptr)
                continue;
            print_node(citer->second);
        }
        cout << *tree_now << endl;
        for (; i < tree_now->children.size(); ++i, ++citer)
        {
            if (citer->second == nullptr)
                continue;
            print_node(citer->second);
        }
    }
    else if (tree->node_type == TAG_LEAF_NODE)// leaf
    {
        shared_ptr<leaf> tree_now = static_pointer_cast<leaf>(tree);

        print_node(nullptr);
        cout << *tree_now << endl;
    }

    return;
}

ostream &operator<<(ostream &os, const categorical_node &node)
{
    cout << "\n\n";

    node.print_space(node.level);
    cout << "parent:";
    if (node.parent.lock() != nullptr)
           cout << node.parent.lock()->attr_name << "::" << node.parent.lock()->child_type(&node) << endl;
    else
        cout << "root" << endl;

    node.print_space(node.level);
    cout << "l:" << node.level << ' ';
    cout << node.node_type << "__" << node.attr_name << endl;

    int num = 0;
    auto riter = node.children.rbegin();
    for (; num < node.children.size() - 1; ++num, ++riter)
    {
        node.print_space(node.level);
        cout << "child" << num + 1 << ':';
        cout << riter->first << endl;
    }
    node.print_space(node.level);
    cout << "child" << num + 1 << ':';
    cout << riter->first;

    return os;
}

ostream &operator<<(ostream &os, const numeric_node &node)
{
    cout << "\n\n";

    node.print_space(node.level);
    cout << "parent:";
    if (node.parent.lock() != nullptr)
           cout << node.parent.lock()->attr_name << "::" << node.parent.lock()->child_type(&node) << endl;
    else
        cout << "root" << endl;

    node.print_space(node.level);
    cout << "l:" << node.level << ' ';
    cout << node.node_type << "__" << node.attr_name << endl;

    node.print_space(node.level);
    cout << "attr_val:" << node.attr_val;

    return os;
}

ostream &operator<<(ostream &os, const leaf &node)
{
    cout << "\n\n";

    node.print_space(node.level);
    cout << "parent:";
       cout << node.parent.lock()->attr_name << "::" << node.parent.lock()->child_type(&node) << endl;

    node.print_space(node.level);
    cout << "l:" << node.level << ' ';
    cout << node.attr_name << endl;
    
    node.print_space(node.level);
    cout << "attr_val:"; 
    cout << node.attr_val << endl;

    node.print_space(node.level);
    cout << "pos_train_num:";
    if (node.pos_train_num != 0)
        cout << node.pos_train_num;
    cout << endl;

    node.print_space(node.level);
    cout << "neg_train_num:";
    if (node.neg_train_num != 0)
        cout << node.neg_train_num;
    cout << endl;

    node.print_space(node.level);
    cout << "pos_test_num:";
    if (node.pos_test_num != 0)
        cout << node.pos_test_num;
    cout << endl;

    node.print_space(node.level);
    cout << "neg_test_num:";
    if (node.neg_test_num != 0)
        cout << node.neg_test_num;

    return os;
}

void tree::prepare_data()
{
    string each_line;
    ifstream fs;

    fs.open(ATTRIBUTES_PATH);
    if (!fs.is_open())
        CRY("attributes file wrong");
    
    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &NUM_NUM_ATTR);
    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &NUM_CTGR_ATTR);
    NUM_ATTR = NUM_NUM_ATTR + NUM_CTGR_ATTR;
    getline(fs, each_line);
    getline(fs, each_line);

    attributes->reserve(NUM_ATTR);
    while (getline(fs, each_line))
    {
        vector<string> line;
        line.reserve(32);
        auto it_begin = each_line.begin();
        auto it_end = find(it_begin, each_line.end(), DLM_ATTR_NAME);
        if (it_end == each_line.end()) // don't read result
            break;

        line.emplace_back(it_begin, it_end);
        it_begin = it_end;
    
        while (it_begin != each_line.end())
        {
            ++it_begin;
            it_end = find(it_begin, each_line.end(), DLM_ATTR_VAL);
            line.emplace_back(it_begin, it_end);
            it_begin = it_end;
        }
        attributes->push_back(line);
    }
    fs.close();

    fs.clear();
    fs.open(TRAINING_DATA_PATH);
    if (!fs.is_open())
        CRY("training examples file wrong");

    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &NUM_TRAINING_DATA);
    exms->attr.reserve(NUM_TRAINING_DATA);
    exms->result.reserve(NUM_TRAINING_DATA);

    int idx_data = 0;
    while (getline(fs, each_line))
    {
        exms->attr[idx_data].reserve(NUM_ATTR);
        auto it_begin = each_line.begin();
        auto it_end = find(it_begin, each_line.end(), DLM_ATTR_VAL);
    
        while (it_end != each_line.end())
        {
            exms->attr[idx_data].emplace_back(it_begin, it_end);
            it_begin = it_end;
            ++it_begin;
            it_end = find(it_begin, each_line.end(), DLM_ATTR_VAL);
        }
        exms->result.emplace_back(it_begin, it_end);
        ++idx_data;
    }
    fs.close();

    fs.clear();
    fs.open(TESTING_DATA_PATH);
    if (!fs.is_open())
        CRY("testing examples file wrong");

    getline(fs, each_line);
    std::sscanf(each_line.c_str(), "%d", &NUM_TESTING_DATA);
    test_exms->attr.reserve(NUM_TESTING_DATA);
    test_exms->result.reserve(NUM_TESTING_DATA);

    idx_data = 0;
    while (getline(fs, each_line))
    {
        test_exms->attr[idx_data].reserve(NUM_ATTR);
        auto it_begin = each_line.begin();
        auto it_end = find(it_begin, each_line.end(), DLM_ATTR_VAL);
    
        while (it_end != each_line.end())
        {
            test_exms->attr[idx_data].emplace_back(it_begin, it_end);
            it_begin = it_end;
            ++it_begin;
            it_end = find(it_begin, each_line.end(), DLM_ATTR_VAL);
        }
        test_exms->result.emplace_back(it_begin, it_end);
        ++idx_data;
    }
    fs.close();
    cout << "----------------------------finish reading data" << endl;
}

} // namespace c4_5

#ifdef UNIT_TESTING

int main()
{
    c4_5::tree t;
    t.build_tree();
    //t.print_tree();
    cout << t.test_tree() << endl;

    return 0;
}

#endif
