#ifndef _YAML_CONSTANTS_H
#define _YAML_CONSTANTS_H

#include "yaml-cpp/yaml.h"
#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;



// YAML file uses the following keys under a root node to describe an
//   experiment.
const string nRovsS = "nRovs";
const string nPOIsS = "nPOIs";
const string couplingS = "coupling";
const string nEpsS = "nEps";
const string nStepsS = "nSteps";
const string staticOrRandomS = "staticOrRandom";
const string typeS = "type";
const string targetsS = "targets";
const string agentsS = "agents";
const string cceaPopS = "ccea_pop";
const string biasStartS = "biasStart";
const string outputS = "output";
const string objectiveS = "objective";
const string obsRS = "obsR";
const string teamS = "T";
const string globalS = "G";
const string teamOS = "UT";
const string trainS = "train";
const string netfileS = "netfile";

// Accessor methods are overloaded to accept vectors -> will nest
const vector<string> xminS = {"world", "xmin"};
const vector<string> yminS = {"world", "ymin"};
const vector<string> xmaxS = {"world", "xmax"};
const vector<string> ymaxS = {"world", "ymax"};

#endif // _YAML_CONSTANTS_H
