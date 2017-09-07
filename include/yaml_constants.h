#ifndef _YAML_CONSTANTS_H
#define _YAML_CONSTANTS_H

#include "yaml-cpp/yaml.h"
#include <iostream>

YAML::Node config = YAML::LoadFile("../input/config.yaml");

const double WORLD_XMIN = config["domain"]["world"]["xmin"].as<double>();
const double WORLD_YMIN = config["domain"]["world"]["ymin"].as<double>();
const double WORLD_XMAX = config["domain"]["world"]["xmax"].as<double>();
const double WORLD_YMAX = config["domain"]["world"]["ymax"].as<double>();

const size_t rovs = config["domain"]["rovers"].as<size_t>();
const size_t nPOIs = config["domain"]["pois"].as<size_t>();
const int coupling = config["domain"]["coupling"].as<int>();

// Team forming
const size_t AGENT_ROVERS = config["domain_agent"]["rovers"].as<size_t>();
const size_t AGENT_POIS = config["domain_agent"]["pois"].as<size_t>();
const int AGENT_COUPLING = config["domain_agent"]["coupling"].as<int>();
const size_t AGENT_EPS = config["scenario"]["agent"]["nEps"].as<size_t>();

// POI seeking
const size_t POI_ROVERS = config["domain_poi"]["rovers"].as<size_t>();
const size_t POI_POIS = config["domain_poi"]["pois"].as<size_t>();
const int POI_COUPLING = config["domain_poi"]["coupling"].as<int>();
const size_t POI_EPS = config["scenario"]["poi"]["nEps"].as<size_t>();

const size_t nSteps = config["scenario"]["nSteps"].as<size_t>();
const size_t nEps = config["scenario"]["nEps"].as<size_t>();
const string evalFunc = config["scenario"]["evalFunc"].as<string>();
const int staticOrRandom = config["scenario"]["staticOrRandom"].as<int>();


const size_t nPop = config["learning"]["ccea"]["nPop"].as<size_t>();

const size_t nInputs = config["learning"]["network"]["nInputs"].as<size_t>();
const size_t nHidden = config["learning"]["network"]["nHidden"].as<size_t>();
const size_t nOutputs = config["learning"]["network"]["nOutputs"].as<size_t>();

void printTestList() {
  YAML::Node node = config["testlist"];
  for (YAML::const_iterator it=node.begin(); it != node.end(); ++it) {
    std::cout << it->as<int>() << "\n";
  }
}

void printTargets() {
  YAML::Node node = config["targets"];

  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    std::cout << "Target" << std::endl;

    YAML::Node tgt = it->as<YAML::Node>();
    for (YAML::const_iterator itr = tgt.begin(); itr != tgt.end(); ++itr) {
      std::cout << itr->as<int>() << std::endl;
    }
  }
}
#endif // _YAML_CONSTANTS_H
