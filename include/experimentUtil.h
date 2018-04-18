/*******************************************************************************
experimentalUtil.h

Experiment utility function header file. Contains definitions for all training
and experiment running utility functions. Includes setting domain, training,
extracting teams from simulation runs.

Authors: Eric Klinkhammer

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*******************************************************************************/

#ifndef _EXPERIMENT_UTIL_H
#define _EXPERIMENT_UTIL_H

#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Eigen>
#include <time.h>
#include <sys/stat.h>

#include "AlignmentAgent.h"

#include "Domains/D.h"
#include "Domains/G.h"
#include "Domains/MultiRover.h"
#include "Domains/Objective.h"
#include "Domains/TeamForming.h"
#include "Domains/TeamObjective.h"

#include "Learning/NeuroEvo.h"

#include "alignments.h"
#include "experimentalSetup.h"
#include "experimentUtil.h"
#include "Statistics.h"
#include "yaml_constants.h"


std::vector<double> testDomain(MultiRover* domain, Objective* o, size_t reps);

// Resets domain according to parameters in YAML Node
// Note that a loop over YAML nodes can repeatedly call this function with
//   the node to reset and train without having to edit constants file.
void setDomain(MultiRover* domain, YAML::Node root);

MultiRover* getDomain(YAML::Node root);

void trainDomain(MultiRover* domain, YAML::Node root, std::string key, std::string topDir,  Objective* o);
// Trains domain for specified number of epochs
void trainDomain(MultiRover*, size_t, bool, int, std::string, std::string,
		 std::string, bool, Objective*);

// Trains domain a single epoch.
void trainDomainOnce(MultiRover*, bool, bool, Objective*);

// Extracts and copies a team of neural nets from a domain
std::vector<NeuralNet> getTeam(MultiRover* domain);

// Creates a MultiRover domain and runs tests according to the specifications
//   and returns the neural networks.
// A domain that is set can be furthered configured (for file output), whereas
//   this function currently only outputs every 20 epochs and returns the
//   networks (one per agent).
std::vector<NeuralNet> trainAndGetTeam(YAML::Node root, std::string, std::string);

AgentType stringToAgentType(std::string);

void configureOutput(MultiRover*, std::string, std::string);

std::vector<NeuroEvo*> readNetworksFromFile(const std::string filename,
					    size_t numIn, size_t numH,
					    size_t numOut, size_t numA,
					    size_t numPop);

vector< NeuralNet* > bestTeamForObjective(YAML::Node expNode, std::string expKey,
					  MultiRover* domain, Objective* o,
					  std::string fileDir);

void configureOutput(MultiRover* domain, string fileDir, string id);
void inspectAgent(MultiRover* domain, VectorXd input);
void trainDomainEps(MultiRover*, YAML::Node, Objective*, size_t);
void trainingCurves(YAML::Node, size_t, size_t, size_t);

inline bool exists_test3 (const std::string& name);

#endif // _EXPERIMENT_UTIL_H
