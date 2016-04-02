#ifndef EVENSONG_TREE
#define EVENSONG_TREE

#include "./stdafx.h"
#include <utility>
#include <tuple>

namespace c4_5
{

using std::ostream;
using std::sprintf;
using std::sscanf;
using std::ifstream;
using std::tuple;
using std::make_tuple;
using std::tie;
using std::weak_ptr;
using std::static_pointer_cast;
using std::abs;
using std::count;
using std::find;
using std::find_if;
using std::find_if_not;
using std::log2;
using std::accumulate;
using std::max_element;
using std::sort;

/*
 * important parameters in decision tree
 */
const string POSITIVE_VAL = ">50K";
const string NEGATIVE_VAL = "<=50K";
const double SINGLE_NODE_THRESHOLD = 0.05;

/*
 * non-important parameters
 */
const string ATTRIBUTES_PATH{"/Users/evensong/ml_data/uci/adult/attr_name"};
const string TRAINING_DATA_PATH{"/Users/evensong/ml_data/uci/adult/training_data"};
const string TESTING_DATA_PATH{"/Users/evensong/ml_data/uci/adult/testing_data"};
const char DLM_ATTR_NAME = ':';
const char DLM_ATTR_VAL = ',';
const string TAG_NUM_ATTR{"continuous"};
const string MISSING_VALUE{"?"};
const char TAG_CTGR_NODE = 'c';
const char TAG_NUM_NODE = 'n';
const char TAG_LEAF_NODE = 'l';
const int LEAF_IDX_ATTR_NAME = -1;

struct node
{
    node(shared_ptr<node> p) : parent(p) {}

    int level = 0;
    char node_type;
    int idx_attr_name = LEAF_IDX_ATTR_NAME; // for leaf node default
    string attr_name;
    weak_ptr<node> parent;

    void print_space(int level) const
    {
        for (int i = 0; i < level; ++i)
            cout << "                                    ";
    }
    virtual string child_type(const node*) const = 0;
    virtual shared_ptr<node> get_proper_child(const string) = 0;
};

struct categorical_node : public node
{
    categorical_node(shared_ptr<node> parent) : node(parent) {}

    std::map<string, shared_ptr<node>> children;

    friend ostream &operator<<(ostream &, const categorical_node &);
    virtual string child_type(const node* child) const
    {
        for (auto p : children)
            if (p.second.get() == child)
                return p.first;
        throw runtime_error("in categorical_node::child_type(), invalid child is given");
    }

    virtual shared_ptr<node> get_proper_child(const string name)
    {
        if (children.find(name) != children.end())
            return children.at(name);
        else if(children.find(string{"?"}) != children.end()) // for single value node
            return children.at(string{"?"});
        else
            throw runtime_error("in categorical_node::get_proper_child(), wrong children name is given:" + name);
    }
};

struct numeric_node : public node
{
    numeric_node(shared_ptr<node> parent) : node(parent) {}

    double attr_val = 0.0;
    shared_ptr<node> lchild;
    shared_ptr<node> rchild;

    friend ostream &operator<<(ostream &, const numeric_node &);
    virtual string child_type(const node* child) const
    {
        if (child == lchild.get())
            return attr_name + "(l)";
        else if (child == rchild.get())
            return attr_name + "(r)";
        else
            throw runtime_error("in numeric_node::child_type(), invalid child is given");
    }

    virtual shared_ptr<node> get_proper_child(const string name)
    {
        double val;
        sscanf(name.c_str(), "%lf", &val);
        if (val <= attr_val)
            return lchild;
        else
            return rchild;
    }
};

struct leaf : public node
{
    leaf(shared_ptr<node> parent) : node(parent) {}

    string attr_val;
    int pos_train_num = 0;
    int neg_train_num = 0;
    int pos_test_num = 0;
    int neg_test_num = 0;

    friend ostream &operator<<(ostream &, const leaf &);
    virtual string child_type(const node* child) const
    {
        throw runtime_error("leaf::child_type() is incorrectly called");
    }

    virtual shared_ptr<node> get_proper_child(const string name)
    {
        throw runtime_error("leaf::get_proper_child() is incorrectly called");
    }
};

struct learning_data
{
    vector<vector<string>> attr;
    vector<string> result;
};

class tree
{
    public:
        tree()
        {
            attributes = make_unique<vector<vector<string>>>();
            exms = make_unique<learning_data>();
            test_exms = make_unique<learning_data>();
            active_attrs = make_unique<vector<int>>();
            prepare_data();
        }
        
        void build_tree()
        {
            auto idx_exm = make_unique<vector<int>>();
            int exm_length = exms->result.size();
            idx_exm->reserve(exm_length);
            for (int i = 0; i < exm_length; ++i)
                idx_exm->push_back(i);
            
            cout << "----------------------------start building tree" << endl;
            decision_tree = create_node(shared_ptr<node>(), *idx_exm, false, 0);
        }

        double test_tree();

        void print_tree() { print_node(decision_tree); }

    private:
        shared_ptr<node> create_node(shared_ptr<node>, const vector<int> &, bool, const int);

        tuple<string, int, bool> comp_best_attr();

        void print_node(shared_ptr<node>);

        void prepare_data();

        bool is_active(int idx_attr_name)
        {
            if (find(active_attrs->begin(), active_attrs->end(), idx_attr_name) == active_attrs->end())
                return true;
            else
                return false;
        }

        int NUM_NUM_ATTR;
        int NUM_CTGR_ATTR;
        int NUM_ATTR;
        int NUM_TRAINING_DATA;
        int NUM_TESTING_DATA;

        shared_ptr<node> decision_tree;
        unique_ptr<vector<vector<string>>> attributes;    
        unique_ptr<learning_data> exms;
        unique_ptr<learning_data> test_exms;
        unique_ptr<vector<int>> active_attrs;
};

} // namespace c4_5

#endif
