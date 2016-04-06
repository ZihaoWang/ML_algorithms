#ifndef EVENSONG_PGM
#define EVENSONG_PGM

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
        size_t add_node(const NODE_INFO_T *info = nullptr)
        {
            nodes.emplace_back(info);
            return nodes.size() - 1;
        }

        void add_edge(const size_t idx_node1, const size_t idx_node2, const EDGE_INFO_T *info = nullptr)
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

        template <typename FN_T>
        void dfs(const size_t idx_start, FN_T fn)
        {
            写DFS
        }

    private :
        
        vector<ud_node<NODE_INFO_T, EDGE_INFO_T>> nodes;
};

} // namespace graph

namespace pgm
{

struct var_node
{
};

struct factor_node
{
};

class factor_graph
{
    public :

    private :
};

} // namespace PGM

#endif
