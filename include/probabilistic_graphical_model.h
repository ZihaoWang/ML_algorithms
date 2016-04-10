#ifndef EVENSONG_PGM
#define EVENSONG_PGM

#include "./graph.h"

namespace pgm
{

using namespace graph;

template <typename EDGE_INFO_T = string> struct mrf_edge;

template<typename VERTEX_INFO_T = string, typename EDGE_T = mrf_edge<>>
struct mrf_vertex : public ud_vertex<VERTEX_INFO_T, EDGE_T>
{
    using ud_vertex<VERTEX_INFO_T, EDGE_T>::visited;
    using ud_vertex<VERTEX_INFO_T, EDGE_T>::edge;

    mrf_vertex(VERTEX_INFO_T *info, const bool is_latent) :
        ud_vertex<VERTEX_INFO_T, EDGE_T>(info), is_latent(is_latent) {}

    mrf_vertex(const double weight, VERTEX_INFO_T *info, const bool is_latent) :
        ud_vertex<VERTEX_INFO_T, EDGE_T>(weight, info), is_latent(is_latent) {}

    bool is_latent;
};

template<typename EDGE_INFO_T>
struct mrf_edge : public ud_edge<EDGE_INFO_T>
{
    //using ud_edge<EDGE_INFO_T>::vertex_next;
    
    const auto &get_vertex_next() 
    {
        return ud_edge<EDGE_INFO_T>::vertex_next;
    }

    mrf_edge(const size_t vertex_now, const size_t vertex_next, EDGE_INFO_T *info) :
        ud_edge<EDGE_INFO_T>(vertex_now, vertex_next, info) {}
    
    mrf_edge(const size_t vertex_now, const size_t vertex_next, const double weight, EDGE_INFO_T *info) :
        ud_edge<EDGE_INFO_T>(weight, vertex_now, vertex_next, info) {}
    
    mrf_edge(const mrf_edge<EDGE_INFO_T> &o) :
        ud_edge<EDGE_INFO_T>(o.weight, o.vertex_next, o.vertex_now, new EDGE_INFO_T(*o.info)) {}
};

template<typename VERTEX_T = mrf_vertex<>, typename EDGE_T = mrf_edge<>>
class mrf : public ud_graph<VERTEX_T, EDGE_T>
{
    public :
        mrf() :
            ud_graph<VERTEX_T, EDGE_T>() {}

        bool is_independent(const size_t vertex1, const size_t vertex2);

        void print() 
        {
            ud_graph<VERTEX_T, EDGE_T>::print();
            cout << "latent:" << endl;
            for (const auto &e : ud_graph<VERTEX_T, EDGE_T>::vertices)
                cout << e->is_latent << '\t';
            cout << endl;
        } 
};

struct var_vertex
{
};

struct factor_vertex
{
};

class factor_graph
{
    public :

    private :
};

} // namespace PGM

#endif
