#include "../include/graph.h"
#include "../include/probabilistic_graphical_model.h"

using namespace graph;
using namespace pgm;

void ud_graph_test()
{
    unique_ptr<ud_graph<>> udg = make_unique<ud_graph<>>();
    size_t last;
    size_t now;
    size_t vertex1;
    last = udg->add_vertex(new ud_vertex<>(new string("a")));
    vertex1 = now = udg->add_vertex(new ud_vertex<>(new string("b")));
    udg->add_edge(new ud_edge<>(last, now, new string("e1")));
    last = now;
    now = udg->add_vertex(new ud_vertex<>(new string("c")));
    udg->add_edge(new ud_edge<>(last, now, new string("e2")));
    now = udg->add_vertex(new ud_vertex<>(new string("d")));
    udg->add_edge(new ud_edge<>(last, now, new string("e3")));
    last = now;
    now = udg->add_vertex(new ud_vertex<>(new string("e")));
    udg->add_edge(new ud_edge<>(last, now, new string("e4")));
    last = vertex1;
    now = udg->add_vertex(new ud_vertex<>(new string("f")));
    udg->add_edge(new ud_edge<>(last, now, new string("e5")));
    last = now;
    now = udg->add_vertex(new ud_vertex<>(new string("g")));
    udg->add_edge(new ud_edge<>(last, now, new string("e6")));

    udg->print();
    cout << "dfs:" << endl;
    udg->dfs(0, [](const size_t idx, ud_vertex<> *vertex){ cout << '\t' << *vertex->info << endl; return false; });
    cout << "bfs:" << endl;
    udg->bfs(0, [](const size_t idx, ud_vertex<> *vertex){ cout << '\t' << *vertex->info << endl; return false; });
    cout << "connectivity:" << endl;
    cout << udg->is_connected(0, 4) << endl;
}

void mrf_test()
{
    unique_ptr<mrf<>> m = make_unique<mrf<>>();
    size_t last;
    size_t now;
    size_t vertex1;
    last = m->add_vertex(new mrf_vertex<>(new string("a"), true));
    vertex1 = now = m->add_vertex(new mrf_vertex<>(new string("b"), true));
    m->add_edge(new mrf_edge<>(last, now, new string("e1")));

    last = now;
    now = m->add_vertex(new mrf_vertex<>(new string("c"), true));
    m->add_edge(new mrf_edge<>(last, now, new string("e2")));
    now = m->add_vertex(new mrf_vertex<>(new string("d"), true));
    m->add_edge(new mrf_edge<>(last, now, new string("e3")));
    last = now;
    now = m->add_vertex(new mrf_vertex<>(new string("e"), true));
    m->add_edge(new mrf_edge<>(last, now, new string("e4")));
    last = vertex1;
    now = m->add_vertex(new mrf_vertex<>(new string("f"), true));
    m->add_edge(new mrf_edge<>(last, now, new string("e5")));
    last = now;
    now = m->add_vertex(new mrf_vertex<>(new string("g"), true));
    m->add_edge(new mrf_edge<>(last, now, new string("e6")));

    m->print();
    cout << "dfs:" << endl;
    m->dfs(0, [](const size_t idx, mrf_vertex<> *vertex){ cout << '\t' << *vertex->info << endl; return false; });
    cout << "bfs:" << endl;
    m->bfs(0, [](const size_t idx, mrf_vertex<> *vertex){ cout << '\t' << *vertex->info << endl; return false; });
    cout << "connectivity:" << endl;
    cout << m->is_connected(0, 4) << endl;
}

int main()
{
    //ud_graph_test();
    mrf_test();
    return 0;
}
