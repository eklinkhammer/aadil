#ifndef _YAML_CONSTANTS_H
#define _YAML_CONSTANTS_H

#include "yaml-cpp/yaml.h"
#include <iostream>

YAML::Node config = YAML::LoadFile("../input/config.yaml");

// Experiment Parameters
const double WORLD_XMIN = config["domain"]["world"]["xmin"].as<double>();
const double WORLD_YMIN = config["domain"]["world"]["ymin"].as<double>();
const double WORLD_XMAX = config["domain"]["world"]["xmax"].as<double>();
const double WORLD_YMAX = config["domain"]["world"]["ymax"].as<double>();
const size_t CCEA_POP   = config["domain"]["ccea_pop"].as<size_t>();

// Training
// Training - Team Formation
const size_t TEAM_ROVERS   = config["training"]["teamOnly"]["nRovs"].as<size_t>();
const size_t TEAM_POIS     = config["training"]["teamOnly"]["nPOIs"].as<size_t>();
const int    TEAM_COUPLING = config["training"]["teamOnly"]["coupling"].as<int>();
const size_t TEAM_EPS      = config["training"]["teamOnly"]["nEps"].as<size_t>();
const size_t TEAM_STEPS    = config["training"]["teamOnly"]["nSteps"].as<size_t>();

// Training - POI
const size_t POI_ROVERS   = config["training"]["POIOnly"]["nRovs"].as<size_t>();
const size_t POI_POIS     = config["training"]["POIOnly"]["nPOIs"].as<size_t>();
const int    POI_COUPLING = config["training"]["POIOnly"]["coupling"].as<int>();
const size_t POI_EPS      = config["training"]["POIOnly"]["nEps"].as<size_t>();
const size_t POI_STEPS    = config["training"]["POIOnly"]["nSteps"].as<size_t>();

// Training - EXPLORING
const size_t EXPLORING_ROVERS   = config["training"]["exploring"]["nRovs"].as<size_t>();
const size_t EXPLORING_POIS     = config["training"]["exploring"]["nPOIs"].as<size_t>();
const int    EXPLORING_COUPLING = config["training"]["exploring"]["coupling"].as<int>();
const size_t EXPLORING_EPS      = config["training"]["exploring"]["nEps"].as<size_t>();
const size_t EXPLORING_STEPS    = config["training"]["exploring"]["nSteps"].as<size_t>();

// Training - Neural Rover
const size_t NEURAL_ROVERS   = config["training"]["neuralRover"]["nRovs"].as<size_t>();
const size_t NEURAL_POIS     = config["training"]["neuralRover"]["nPOIs"].as<size_t>();
const int    NEURAL_COUPLING = config["training"]["neuralRover"]["coupling"].as<int>();
const size_t NEURAL_EPS      = config["training"]["neuralRover"]["nEps"].as<size_t>();
const size_t NEURAL_STEPS    = config["training"]["neuralRover"]["nSteps"].as<size_t>();

// Testing
// Testing - Optimal Team Only
const int    ONLY_T_COUPLING = config["tests"]["teamOnly"]["coupling"].as<int>();
const size_t ONLY_T_EPS      = config["tests"]["teamOnly"]["nEps"].as<size_t>();
const size_t ONLY_T_STEPS    = config["tests"]["teamOnly"]["nSteps"].as<size_t>();
YAML::Node   ONLY_T_TARGETS  = config["tests"]["teamOnly"]["targets"].as<YAML::Node>();
YAML::Node   ONLY_T_ROVERS   = config["tests"]["teamOnly"]["agents"].as<YAML::Node>();
					    
// Testing - Optimal POI ONly
const int    ONLY_P_COUPLING = config["tests"]["POIOnly"]["coupling"].as<int>();
const size_t ONLY_P_EPS      = config["tests"]["POIOnly"]["nEps"].as<size_t>();
const size_t ONLY_P_STEPS    = config["tests"]["POIOnly"]["nSteps"].as<size_t>();
YAML::Node   ONLY_P_TARGETS  = config["tests"]["POIOnly"]["targets"].as<YAML::Node>();
YAML::Node   ONLY_P_ROVERS   = config["tests"]["POIOnly"]["agents"].as<YAML::Node>();


// Testing - Team Then POI
const int    P_AND_T_COUPLING = config["tests"]["teamThenPOI"]["coupling"].as<int>();
const size_t P_AND_T_EPS      = config["tests"]["teamThenPOI"]["nEps"].as<size_t>();
const size_t P_AND_T_STEPS    = config["tests"]["teamThenPOI"]["nSteps"].as<size_t>();
YAML::Node   P_AND_T_TARGETS  = config["tests"]["teamThenPOI"]["targets"].as<YAML::Node>();
YAML::Node   P_AND_T_ROVERS   = config["tests"]["teamThenPOI"]["agents"].as<YAML::Node>();

#endif // _YAML_CONSTANTS_H
