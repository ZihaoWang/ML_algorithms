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

template <typename EDGE_INFO_T = string> struct d_edge;

template <typename VERTEX_INFO_T = string, typename EDGE_INFO_T = string>
struct d_vertex
{
};

template <typename EDGE_INFO_T>
struct d_edge
{
    unique_ptr<EDGE_INFO_T> info;
};

template <typename VERTEX_T = d_vertex<>, typename EDGE_T = d_edge<>>
class d_graph
{
    public :
        void add_vertex();


    private :
};

// undirected graph

/*
 * for undirected graph, EDGE_INGO_T should offer valid copy-constructor
 */
template <typename EDGE_INFO_T = string> struct ud_edge;

template <typename VERTEX_INFO_T = string, typename EDGE_T = ud_edge<>>
struct ud_vertex
{
    ud_vertex(VERTEX_INFO_T *info) :
        visited(false), weight(0.0), index(numeric_limits<size_t>::max()), info(info) {}

    ud_vertex(const double weight, VERTEX_INFO_T *info) :
        visited(false), weight(weight), index(numeric_limits<size_t>::max()), info(info) {}

    bool visited;
    size_t index;
    double weight;
    unique_ptr<VERTEX_INFO_T> info;
    vector<unique_ptr<EDGE_T>> edge;
};

template <typename EDGE_INFO_T>
struct ud_edge
{
    ud_edge(const size_t vertex_now, const size_t vertex_next, EDGE_INFO_T *info) :
        visited(false), weight(0.0), vertex_now(vertex_now), vertex_next(vertex_next), info(info) {}
    
    ud_edge(const double weight, const size_t vertex_now, const size_t vertex_next, EDGE_INFO_T *info) :
        visited(false), weight(weight), vertex_now(vertex_now), vertex_next(vertex_next), info(info) {}
    
    ud_edge(const ud_edge &o) :
        visited(false), weight(o.weight), vertex_now(o.vertex_next), vertex_next(o.vertex_now),
        info(new EDGE_INFO_T(*o.info)) {}

    bool visited;
    double weight;
    size_t vertex_now;
    size_t vertex_next;
    unique_ptr<EDGE_INFO_T> info;
};

template<typename VERTEX_T = ud_vertex<>, typename EDGE_T = ud_edge<>>
class ud_graph
{
    public :
        ud_graph() :
            has_weight(false) {}

        size_t add_vertex(VERTEX_T *vertex) 
        { 
            if (std::abs(vertex->weight) > numeric_limits<double>::epsilon())
                has_weight = true;
            vertex->index = vertices.size();
            vertices.emplace_back(vertex);
            return vertices.size() - 1;
        }

        void add_edge(EDGE_T *edge);

        /*
         * for dsf(), bfs(), is_connected()
         * FN_T should be a function whose prototype is: 
         * bool fn(const size_t, VERTEX_T *)
         *
         * for fn():
         * if return value is true, search would terminate
         * arg1: index of vertex
         * arg2: reference of vertex
         */
        template <typename FN_T>
        bool dfs(const size_t idx_begin, FN_T fn) const;

        template <typename FN_T>
        bool bfs(const size_t idx_begin, FN_T fn) const;

        bool is_connected(const size_t idx_vertex1, const size_t idx_vertex2) const
        {
            return bfs(idx_vertex1, [idx_vertex2](const size_t idx_vertex1, VERTEX_T *vertex){ return idx_vertex1 == idx_vertex2 ? true : false; });
        }

        template <typename FN_T>
        bool is_connected(const size_t idx_vertex1, const size_t idx_vertex2, FN_T conn_pred) const
        {
            return bfs(idx_vertex1, conn_pred);
        }

        // to be implemented
        vector<unique_ptr<EDGE_T>> *dijkstra(const size_t idx_vertex1, const size_t idx_vertex2);

        void print() const;

        const auto &get_vertices() const
        { 
            return vertices; 
        }

    protected :
        bool has_weight;
        vector<unique_ptr<VERTEX_T>> vertices;
        vector<unique_ptr<EDGE_T>> path;
};

template<typename VERTEX_T, typename EDGE_T>
void ud_graph<VERTEX_T, EDGE_T>::add_edge(EDGE_T *edge)
{
    if (edge->vertex_now >= vertices.size() || edge->vertex_next >= vertices.size())
        CRY();
    if (std::abs(edge->weight) > numeric_limits<double>::epsilon())
        has_weight = true;

    auto &tmp = vertices[edge->vertex_now]->edge;
    if (tmp.empty())
    {
        tmp.emplace_back(edge);
        vertices[edge->vertex_next]->edge.emplace_back(new EDGE_T(*edge));
    }
    else
    {
        bool edge_existed = std::find_if(tmp.begin(), tmp.end(), [&edge](const unique_ptr<EDGE_T> &e){ return e->vertex_next == edge->vertex_next; }) != tmp.end();
        if (!edge_existed)
        {
            tmp.emplace_back(edge);
            vertices[edge->vertex_next]->edge.emplace_back(new EDGE_T(*edge));
        }
    }
}

template<typename VERTEX_T, typename EDGE_T>
template <typename FN_T>
bool ud_graph<VERTEX_T, EDGE_T>::dfs(const size_t idx_begin, FN_T fn) const
{
    if (idx_begin < 0 || idx_begin >= vertices.size())
        CRY();

    stack<pair<size_t, size_t>> stk;
    if (fn(idx_begin, vertices[idx_begin].get()))
        return true;
    vertices[idx_begin]->visited = true;
    stk.push(make_pair(idx_begin, 0));

    while (!stk.empty())
    {
        auto &vertex_now = *vertices[stk.top().first];
        size_t idx_edge = stk.top().second;
        for (; idx_edge < vertex_now.edge.size(); ++idx_edge)
        {
            size_t idx_next = vertex_now.edge[idx_edge]->vertex_next;
            if (!vertices[idx_next]->visited)
            {
                if (fn(idx_next, vertices[idx_next].get()))
                    return true;
                vertices[idx_next]->visited = true;
                stk.top().second = idx_edge + 1;
                stk.push(make_pair(idx_next, 0));
                break;
            }
        }
        if (idx_edge >= vertex_now.edge.size())
            stk.pop();
    }

    for (const auto &e : vertices)
        e->visited = false;
    return false;
}

template<typename VERTEX_T, typename EDGE_T>
template <typename FN_T>
bool ud_graph<VERTEX_T, EDGE_T>::bfs(const size_t idx_begin, FN_T fn) const
{
    if (idx_begin < 0 || idx_begin >= vertices.size())
        CRY();

    queue<size_t> que;
    que.push(idx_begin);
    while (!que.empty())
    {
        size_t idx_now = que.front();
        auto &vertex_now = *vertices[idx_now];
        que.pop();
        if (fn(idx_now, vertices[idx_now].get()))
            return true;
        vertices[idx_now]->visited = true;

        for (const auto &e : vertex_now.edge)
            if (!vertices[e->vertex_next]->visited)
                que.push(e->vertex_next);
    }

    for (const auto &e : vertices)
        e->visited = false;
    return false;
}

template<typename VERTEX_T, typename EDGE_T>
vector<unique_ptr<EDGE_T>> *ud_graph<VERTEX_T, EDGE_T>::dijkstra(const size_t idx_vertex1, const size_t idx_vertex2)
{
    path.clear();

    return path;
}

template<typename VERTEX_T, typename EDGE_T>
void ud_graph<VERTEX_T, EDGE_T>::print() const
{
    for (int i = 0; i < vertices.size(); ++i)
    {
        cout << "vertex " << i << ", info: " << *vertices[i]->info << endl;
        cout << "has edges to:" << endl;
        for (const auto &e : vertices[i]->edge)
            cout << "\tto vertex: " << e->vertex_next << ", info: " << *e->info << endl;
    }
}

} // namespace graph

#endif
