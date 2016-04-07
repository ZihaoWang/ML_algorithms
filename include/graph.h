#ifndef EVENSONG_GRAPH
#define EVENSONG_GRAPH

#include "./stdafx.h"

namespace graph
{

// directed graph

template <typename EDGE_INFO_T>
struct d_edge
{
    unique_ptr<INFO_T> info;
};

template <typename NODE_INFO_T, typename EDGE_INFO_T>
class d_graph
{
    public :
        void add_node();


    private :
};

// undirected graph

template <typename NODE_INFO_T, typename EDGE_INFO_T>
struct ud_node
{
    ud_node(const NODE_INFO_T *info) :
        visited{false}, info{info} {}

    bool visited;
    unique_ptr<NODE_INFO_T> info;
    vector<ud_edge<EDGE_INFO_T>> edge;
};

template <typename EDGE_INFO_T>
struct ud_edge
{
    ud_edge(const size_t idx_node, const EDGE_INFO_T *info) :
        idx_node{idx_node}, info{info} {}
    
    size_t idx_node;
    unique_ptr<EDGE_INFO_T> info;
};

template<typename NODE_INFO_T, typename EDGE_INFO_T>
class ud_graph
{
    public :
        void add_node(const NODE_INFO_T *info = nullptr) { nodes.emplace_back(info); }

        void add_edge(const size_t idx_node1, const size_t idx_node2, const EDGE_INFO_T *info = nullptr);

        template <typename FN_T>
        void dfs(const size_t idx_begin, FN_T fn);

        template <typename FN_T>
        void bfs(const size_t idx_begin, FN_T fn);

        void print();

        const auto &get_nodes() { return nodes; }

    private :
        vector<ud_node<NODE_INFO_T, EDGE_INFO_T>> nodes;
};

template<typename NODE_INFO_T, typename EDGE_INFO_T>
void ud_graph<NODE_INFO_T, EDGE_INFO_T>::add_edge(const size_t idx_node1, const size_t idx_node2, const EDGE_INFO_T *info)
{
    if (idx_node1 >= nodes.size())
        CRY();
    if (idx_node2 >= nodes.size())
        CRY();

    if (nodes[idx_node1].edge.empty())
        nodes[idx_node1].edge.emplace_back(idx_node2, info);
    else
        if (std::find_if(nodes[idx_node1].edge.begin(), nodes[idx_node1].edge.end(), 
                    [idx_node1](const ud_edge<EDGE_INFO_T> &e){ return e.idx_node == idx_node1 ? true : false;})
                == nodes[idx_node1].edge.end())
            nodes[idx_node1].edge.emplace_back(idx_node2, info);

    if (nodes[idx_node2].edge.empty())
        nodes[idx_node2].edge.emplace_back(idx_node1, info);
    else
        if (std::find_if(nodes[idx_node2].edge.begin(), nodes[idx_node2].edge.end(), 
                    [idx_node2](const ud_edge<EDGE_INFO_T> &e){ return e.idx_node == idx_node2 ? true : false;})
                == nodes[idx_node2].edge.end())
            nodes[idx_node2].edge.emplace_back(idx_node1, info);
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
template <typename FN_T>
void ud_graph<NODE_INFO_T, EDGE_INFO_T>::dfs(const size_t idx_begin, FN_T fn)
{
    if (idx_begin < 0 || idx_begin >= nodes.size())
        CRY();

    stack<pair<size_t, size_t>> stk;
    stk.push(make_pair(idx_begin, 0));
    nodes[idx_begin].visited = true;
    while (!stk.empty())
    {
        auto node_now = nodes[stk.top().first];
        size_t idx_edge = stk.top().second;
        for (; idx_edge < node_now.edge.size(); ++idx_edge)
        {
            size_t idx_next = node_now.edge[idx_edge].idx_node;
            if (!nodes[idx_next].visited)
            {
                fn(nodes[idx_next]);
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
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
template <typename FN_T>
void ud_graph<NODE_INFO_T, EDGE_INFO_T>::bfs(const size_t idx_begin, FN_T fn)
{
    if (idx_begin < 0 || idx_begin >= nodes.size())
        CRY();

    重写bfs
    stack<pair<size_t, size_t>> stk;
    stk.push(make_pair(idx_begin, 0));
    nodes[idx_begin].visited = true;
    while (!stk.empty())
    {
        auto node_now = nodes[stk.top().first];
        size_t idx_edge = stk.top().second;
        for (; idx_edge < node_now.edge.size(); ++idx_edge)
        {
            size_t idx_next = node_now.edge[idx_edge].idx_node;
            if (!nodes[idx_next].visited)
            {
                fn(nodes[idx_next]);
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
}

template<typename NODE_INFO_T, typename EDGE_INFO_T>
void ud_graph<NODE_INFO_T, EDGE_INFO_T>::print()
{
    for (int i = 0; i < nodes.size(); ++i)
    {
        cout << "node " << i << "has edges to:" << endl;
        for (const auto &e : nodes[i].edge)
            cout << "\tnode " << e.idx_node << endl;
    }
}

} // namespace graph

#endif
