//
// Created by Baoxu Shi on 6/11/15.
//

#ifndef GBPEDIA_OPTS_H
#define GBPEDIA_OPTS_H

#include <boost/program_options.hpp>
  class opts {
    boost::program_options::options_description desc;

  private:
    std::string node_filepath;
    std::string edge_filepath;
    std::string type_filepath;
    int port = 8964;
    bool is_directed;
    int nworker = 10;
      unsigned int rel_type = 671;

  public:
    opts();

    bool parse(int argc, const char* argv[]);

  public:

      const unsigned int &getRel_type() const {
        return rel_type;
      }

    const std::string &getNode_filepath() const {
      return node_filepath;
    }

    const std::string &getEdge_filepath() const {
      return edge_filepath;
    }

    const std::string &getType_filepath() const {
      return type_filepath;
    }

    int getPort() const {
      return port;
    }

    bool getIs_directed() const {
      return is_directed;
    }

    int getNworker() const {
      return nworker;
    }

  };



#endif //GBPEDIA_OPTS_H
