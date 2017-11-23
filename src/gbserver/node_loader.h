//
// Created by Baoxu Shi on 6/12/15.
//

#ifndef GBPEDIA_NODE_H
#define GBPEDIA_NODE_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

template <typename T> class node_loader {

public:
  typedef T value_type;

private:
  std::vector<value_type> node_map;
  unsigned int max_id;
  unsigned int min_id;

public:

  node_loader(std::string node_filepath) noexcept : max_id(0), min_id(0), node_map(std::vector<value_type>()) {

    std::fstream fin(node_filepath, std::fstream::in);

    value_type val;
    unsigned int id;

    while(!fin.eof()) {
      fin >> id;
      fin.get(); // remove separator
      fin >> val;

      if (id >= node_map.size()) { /* Enlarge node_map so it can contain certain number of nodes */
        node_map.resize(id + 1u, "");
      }
      node_map[id] = val;

      max_id = std::max(max_id, id);
      min_id = std::min(min_id, id);
    }

    fin.close();
  };

  bool exists(unsigned int id) const noexcept {
    return (id <= max_id) && (id >= min_id);
  }

  unsigned int getMax_id() const noexcept {
    return max_id;
  }

  unsigned int getMin_id() const noexcept {
    return min_id;
  }

  value_type get_value(unsigned int id) const {
    if (id > getMax_id()) {
      throw std::runtime_error(std::to_string(id) + " is larger than max id in nodelist.");
    }
    return node_map.at(id);
  }

};


#endif //GBPEDIA_NODE_H
