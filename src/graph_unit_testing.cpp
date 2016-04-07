#include "../include/graph.h"
#include "../include/probabilistic_graphical_model.h"

int main()
{
    unique_ptr<pgm::mrf<string, string>> udg = make_unique<pgm::mrf<string, string>>();
    size_t last;
    size_t now;
    size_t node1;
    last = udg->add_node(new string("a"));
    node1 = now = udg->add_node(new string("b"));
    udg->add_edge(last, now, new string("e1"));
    last = now;
    now = udg->add_node(new string("c"));
    udg->add_edge(last, now, new string("e2"));
    now = udg->add_node(new string("d"));
    udg->add_edge(last, now, new string("e3"));
    last = now;
    now = udg->add_node(new string("e"));
    udg->add_edge(last, now, new string("e4"));
    last = node1;
    now = udg->add_node(new string("f"));
    udg->add_edge(last, now, new string("e5"));
    last = now;
    now = udg->add_node(new string("g"));
    udg->add_edge(last, now, new string("e6"));

    udg->print();
    cout << "dfs:" << endl;
    udg->dfs(0, [](const size_t idx, graph::ud_node<string, string> &node){ cout << '\t' << *node.info << endl; return false; });
    cout << "bfs:" << endl;
    udg->bfs(0, [](const size_t idx, graph::ud_node<string, string> &node){ cout << '\t' << *node.info << endl; return false; });
    cout << "connectivity:" << endl;
    cout << udg->is_connected(0, 4) << endl;


    return 0;
}
