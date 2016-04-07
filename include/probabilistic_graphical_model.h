#ifndef EVENSONG_PGM
#define EVENSONG_PGM

#include "./graph.h"

namespace pgm
{

using namespace graph;

template<typename NODE_INFO_T, typename EDGE_INFO_T>
struct mrf_node : public ud_node<NODE_INFO_T, EDGE_INFO_T>
{
};

template<typename EDGE_INFO_T>
struct mrf_edge : public ud_edge<EDGE_INFO_T>
{
};

template<typename NODE_INFO_T, typename EDGE_INFO_T>
class mrf : public ud_graph<NODE_INFO_T, EDGE_INFO_T>
{
    public :
        void ff() {this->print();} 
};

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
