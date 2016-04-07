#ifndef EVENSONG_GRAPH
#define EVENSONG_GRAPH

#include "./stdafx.h"
#include <stack>
#include <queue>

namespace graph
{

using std::stack;
using std::queue;
using std::priority_queue;

// directed graph

template <typename EDGE_INFO_T>
struct d_edge
{
    unique_ptr<EDGE_INFO_T> info;
};

template <typename NODE_INFO_T, typename EDGE_INFO_T>
struct d_node
{
};

template <typename NODE_INFO_T, typename EDGE_INFO_T>
class d_graph
{
    public :
        void add_node();


    private :
};

// undirected graph

/*
 * for undirected graph, EDGE_INGO_T should offer valid copy-constructor
 */
template <typename EDGE_INFO_T>
struct ud_edge
{
    ud_edge(const size_t idx_node, const double weight, EDGE_INFO_T *info) :
        idx_node(idx_node), weight(weight), info(info) {}
    
    ud_edge(ud_edge &&old) noexcept :
        idx_node(old.idx_node), weight(old.weight), info(move(old.info)) {}

    size_t idx_node;
    double weight;
    unique_ptr<EDGE_INFO_T> info;
};

template <typename NODE_INFO_T, typename EDGE_INFO_T>
struct ud_node
{
    ud_node(const double weight, NODE_INFO_T *info) :
        visited(false), weight(weight), info(info) {}

    ud_node(ud_node &&old) noexcept :
        visited(old.visited), weight(old.weight), info(move(old.info)), edge(move(old.edge)) {}

    bool visited;
    double weight;
    unique_ptr<NODE_INFO_T> info;
    vector<ud_edge<EDGE_INFO_T>> edge;
};

template<typename NODE_INFO_T, typename EDGE_INFO_T>
class ud_graph
{
    public :
        ud_graph() :
            has_weight(false) {}

        size_t add_node(NODE_INFO_T *info = nullptr, const double weight = 0.0) 
        { 
            if (std::abs(weight) > numeric_limits<double>::epsilon())
                has_weight = true;
            nodes.emplace_back(weight, info);
            return nodes.size() - 1;
        }

        void add_edge(const size_t idx_node1, const size_t idx_node2, EDGE_INFO_T *info = nullptr, const double weight = 0.0);

        /*
         * for dsf(), bfs(), is_connected()
         * FN_T should be a function whose prototype is: 
         * bool fn(ud_node<const size_t, ud_node<NODE_INFO_T, EDGE_INFO_T>> &)
         *
         * for fn():
         * if return value is true, search would terminate
         * arg1: index of node
         * arg2: reference of node
         */
        template <typename FN_T>
        bool dfs(const size_t idx_begin, FN_T fn);

        template <typename FN_T>
        bool bfs(const size_t idx_begin, FN_T fn);

        bool is_connected(const size_t idx_node1, const size_t idx_node2)
        {
            return bfs(idx_node1, [idx_node2](const size_t idx_node1, ud_node<NODE_INFO_T, EDGE_INFO_T> &node){ return idx_node1 == idx_node2 ? true : false; });
        }

        template <typename FN_T>
        bool is_connected(const size_t idx_node1, const size_t idx_node2, FN_T conn_pred)
        {
            return bfs(idx_node1, conn_pred);
        }

        // to be implemented
        const vector<ud_edge<EDGE_INFO_T>> *dijkstra(const size_t idx_node1, const size_t idx_node2);

        void print();

        const auto &get_nodes() { return nodes; }

    private :
        bool has_weight;
        vector<ud_node<NODE_INFO_T, EDGE_INFO_T>> nodes;
        vector<ud_edge<EDGE_INFO_T>> path;
};

template<typename NODE_INFO_T, typename EDGE_INFO_T>
void ud_graph<NODE_INFO_T, EDGE_INFO_T>::add_edge(const size_t idx_node1, const size_t idx_node2, 
        EDGE_INFO_T *info1, const double weight)
{
    if (idx_node1 >= nodes.size())
        CRY();
    if (idx_node2 >= nodes.size())
        CRY();
    EDGE_INFO_T *info2 = new EDGE_INFO_T(*info1);

    if (std::abs(weight) > numeric_limits<double>::epsilon())
        has_weight = true;
    if (nodes[idx_node1].edge.empty())
        nodes[idx_node1].edge.emplace_back(idx_node2, weight, info1);
    else
        if (std::find_if(nodes[idx_node1].edge.begin(), nodes[idx_node1].edge.end(), 
                    [idx_node1](const ud_edge<EDGE_INFO_T> &e){ return e.idx_node == idx_node1 ? true : false;})
                == nodes[idx_node1].edge.end())
            nodes[idx_node1].edge.emplace_back(idx_node2, weight, info1);

    if (nodes[idx_node2].edge.empty())
        nodes[idx_node2].edge.emplace_back(idx_node1, weight, info2);
    else
        if (std::find_if(nodes[idx_node2].edge.begin(), nodes[idx_node2].edge.end(), 
                    [idx_node2](const ud_edge<EDGE_INFO_T> &e){ return e.idx_node == idx_node2 ? true : false;})
                == nodes[idx_node2].edge.end())
            nodes[idx_node2].edge.emplace_back(idx_node1, weight, info2);
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
template <typename FN_T>
bool ud_graph<NODE_INFO_T, EDGE_INFO_T>::dfs(const size_t idx_begin, FN_T fn)
{
    if (idx_begin < 0 || idx_begin >= nodes.size())
        CRY();

    stack<pair<size_t, size_t>> stk;
    if (fn(idx_begin, nodes[idx_begin]))
        return true;
    nodes[idx_begin].visited = true;
    stk.push(make_pair(idx_begin, 0));

    while (!stk.empty())
    {
        auto &node_now = nodes[stk.top().first];
        size_t idx_edge = stk.top().second;
        for (; idx_edge < node_now.edge.size(); ++idx_edge)
        {
            size_t idx_next = node_now.edge[idx_edge].idx_node;
            if (!nodes[idx_next].visited)
            {
                if (fn(idx_next, nodes[idx_next]))
                    return true;
                nodes[idx_next].visited = true;
                stk.top().second = idx_edge + 1;
                stk.push(make_pair(idx_next, 0));
                break;
            }
        }
        if (idx_edge >= node_now.edge.size())
            stk.pop();
    }

    for (auto &e : nodes)
        e.visited = false;
    return false;
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
template <typename FN_T>
bool ud_graph<NODE_INFO_T, EDGE_INFO_T>::bfs(const size_t idx_begin, FN_T fn)
{
    if (idx_begin < 0 || idx_begin >= nodes.size())
        CRY();

    queue<size_t> que;
    que.push(idx_begin);
    while (!que.empty())
    {
        size_t idx_now = que.front();
        auto &node_now = nodes[idx_now];
        que.pop();
        if (fn(idx_now, node_now))
            return true;
        nodes[idx_now].visited = true;

        for (const auto &e : node_now.edge)
            if (!nodes[e.idx_node].visited)
                que.push(e.idx_node);
    }

    for (auto &e : nodes)
        e.visited = false;
    return false;
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
const vector<ud_edge<EDGE_INFO_T>> *ud_graph<NODE_INFO_T, EDGE_INFO_T>::dijkstra(const size_t idx_node1, const size_t idx_node2)
{
    path.clear();

    return path;
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
void ud_graph<NODE_INFO_T, EDGE_INFO_T>::print()
{
    for (int i = 0; i < nodes.size(); ++i)
    {
        cout << "node " << i << ", info: " << *nodes[i].info << endl;
        cout << "has edges to:" << endl;
        for (const auto &e : nodes[i].edge)
            cout << "\tnode " << e.idx_node << ", info: " << *e.info << endl;
    }
}

} // namespace graph

#endif
