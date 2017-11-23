//
// Created by Baoxu Shi on 6/13/15.
//

#ifndef GBPEDIA_EDGE_LIST_H
#define GBPEDIA_EDGE_LIST_H

#include <vector>
#include <fstream>
#include <algorithm>
#include <utility>
#include <set>

class edge_list {
  /* Edge pairs are <node,rel>, the other endpoint is the index of the container */
  std::set< std::pair<uint, uint> > forward; /* Directed edge*/
  std::set< std::pair<uint, uint> > backward; /* Reversed edge for mimicking undirected graph */
  std::set<uint> in_neighbors;
  std::set<uint> out_neighbors;
  std::set<uint> neighbors;

public:

  edge_list() : forward(std::set< std::pair<uint, uint> >()),
                backward(std::set< std::pair<uint, uint> >()){}

  inline void connect_to(uint target, uint rel) {
    forward.insert(std::pair<uint, uint>(target, rel));
    out_neighbors.insert(target);
    neighbors.insert(target);
  }
  inline void connected_by(uint source, uint rel) {
    backward.insert(std::pair<uint, uint>(source, rel));
    in_neighbors.insert(source);
    neighbors.insert(source);
  }
  inline void disconnect_to(uint target, uint rel) {
    forward.erase(std::pair<uint, uint>(target, rel));
    //TODO: delete neighbor from the set
  }
  inline void disconnected_by(uint source, uint rel) {
    backward.erase(std::pair<uint, uint>(source, rel));
    //TODO: delete neighbor from the set
  }

  std::set< std::pair<uint, uint> >& get_forward() noexcept {return forward;}
  std::set< std::pair<uint, uint> >& get_backward() noexcept {return backward;}

  const std::set<uint> &get_in_neighbors()  noexcept { return in_neighbors; }

  const std::set<uint> &get_out_neighbors()  noexcept { return out_neighbors; }

  const std::set<uint> &get_neighbors() noexcept {
    return neighbors;
  }

  inline size_t get_in_deg() noexcept {
    return in_neighbors.size();
  }

  inline size_t get_out_deg() noexcept {
    return out_neighbors.size();
  }

  inline size_t get_deg() noexcept {
    return neighbors.size();
  }

};


#endif //GBPEDIA_EDGE_LIST_H
