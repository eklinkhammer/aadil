/*******************************************************************************
Main file

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

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "experimentUtil.h"

#include "Domains/Objective.h"
#include "Domains/G.h"
#include "Domains/TeamForming.h"
#include "Domains/TeamObjective.h"

#include "Agents/AlignmentAgent.h"
#include "alignments.h"
#include "Statistics.h"

#include <time.h>

using std::vector ;
using std::string ;
using namespace Eigen ;

int readTrialNum() {
  int trialNum;
  std:: cout << "Please enter trial number [NOTE: no checks enabled to prevent"
	     << " overwriting existing files, user must make sure trial number "
	     << "is unique]: ";
  std::cin >> trialNum;
  return trialNum;
}

void makeDir(std::string dir) {
  std::string command = "mkdir -p " + dir;
  system(command.c_str());
}

// void printConfig(std::string fileName) {
//   std::ofstream stream;
//   stream.open(fileName.c_str(), std::ios::app);
//   stream << config << std::endl;
//   stream.close();
// }

AgentType stringToAgentType(std::string type) {
  if      (type == "A") return AgentType::A;
  else if (type == "P") return AgentType::P;
  else if (type == "R") return AgentType::R;
  else if (type == "M") return AgentType::M;
  else if (type == "E") return AgentType::E;
  else if (type == "C") return AgentType::C;
  else                  return AgentType::R;
}

void trainDomain(MultiRover* domain, size_t epochs, bool output, int outputPeriod,
		 string exp, string topDir, string id, bool init, Objective* o) {
  for (size_t n = 0; n < epochs; n++) {
    if (output && (n % outputPeriod == 0 || n == epochs - 1)) {
      std::cout << "Training " << id << " Episode " << n << "...";
      domain->setVerbose(true);
    } else {
      domain->setVerbose(false);
    }
    if (n == epochs - 1) {
      configureOutput(domain, topDir, id);
    }

    trainDomainOnce(domain, (n==0), init, o);
  }
}

void trainDomainOnce(MultiRover* domain, bool evolve, bool init, Objective* o) {
  domain->EvolvePolicies(evolve);
  if (init) {
    domain->InitialiseEpoch();
  }
  domain->ResetEpochEvals();
  domain->SimulateEpoch(true, o);
}

void testDomainOnce(MultiRover* domain, bool output) {
  domain->setVerbose(output);
  domain->InitialiseEpoch();
  domain->ResetEpochEvals();
  //  domain->SimulateEpoch(false);
}

MultiRover* getDomain(YAML::Node root) {
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);

  size_t cceaPop = size_tFromYAML(root, cceaPopS);

  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};
  
  MultiRover* domain = new MultiRover(world, nSteps, cceaPop, nPOIs, Fitness::G, nRovs, coupling, t);
  
  int staticOrRandom = intFromYAML(root, staticOrRandomS);
  if (staticOrRandom == 0) {
    YAML::Node targetNode = nodeFromYAML(root, targetsS);
    YAML::Node agentNode = nodeFromYAML(root, agentsS);
    vector<Target> targets = targetsFromYAML(targetNode);
    vector<Vector2d> initXYs = vector2dsFromYAML(agentNode);
    domain->InitialiseEpochFromVectors(targets, initXYs);
  }

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain->setBias(false);
  }

  return domain;
}

// Initialises a MultiRover domain according to the specifications in the node
void setDomain(MultiRover* domain, YAML::Node root) {
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);
  
  domain->setNRovers(nRovs);
  domain->setNPOIs(nPOIs);
  domain->setNSteps(nSteps);
  domain->setCoupling(coupling);
  domain->setType(t);

  domain->initRovers();
  
  int staticOrRandom = intFromYAML(root, staticOrRandomS);
  if (staticOrRandom == 0) {
    YAML::Node targetNode = nodeFromYAML(root, targetsS);
    YAML::Node agentNode = nodeFromYAML(root, agentsS);
    vector<Target> targets = targetsFromYAML(targetNode);
    vector<Vector2d> initXYs = vector2dsFromYAML(agentNode);
  }

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain->setBias(false);
  }
}

std::vector<NeuralNet> getTeam(MultiRover* domain) {
  std::vector<NeuralNet*> myNets = domain->getNNTeam();
  std::vector<NeuralNet> returnNets;
  
  for (auto& net : myNets) {
    NeuralNet newNet(net->getNI(), net->getNO(), net->getNH());
    newNet.SetWeights(net->GetWeightsA(), net->GetWeightsB());
    returnNets.push_back(newNet);
  }
  return returnNets;
}

void trainDomain(MultiRover* domain, YAML::Node root, std::string key, std::string topDir, Objective* o) {
  size_t nEps = size_tFromYAML(root, nEpsS);
  int output = intFromYAML(root, outputS);
  bool toOutput = (output == 1);
  
  string type = stringFromYAML(root, typeS);
  // AgentType t = stringToAgentType(type);

  int staticOrRandom = intFromYAML(root, staticOrRandomS);
  bool random = staticOrRandom == 1;
  
  trainDomain(domain, nEps, toOutput, 20, type, topDir, key, random, o);
}

void configureOutput(MultiRover* domain, string fileDir, string id) {
  string resultFile = fileDir + "/" + id + "_results";
  string trajFile   = fileDir + "/" + id + "_trajectory";
  string poiFile    = fileDir + "/" + id + "_POIs";
  string choiceFile = fileDir + "/" + id + "_choices";
  domain->OutputPerformance(resultFile);
  domain->OutputTrajectories(trajFile, poiFile, choiceFile);
}

void inspectAgent(MultiRover* domain, VectorXd input) {
  vector<Agent*> agents = domain->getAgents();

  //for (const auto& agent : agents) {
    vector<State> states = agents[0]->allNextGetNextState(input);
    std::cout << "Current State" << std::endl;
    std::cout << agents[0]->getCurrentState() << std::endl;
    //for (const auto& state : states) {
      std::cout << states[0] << std::endl;
      //}
    //}
}
int main() {
  time_t time_before, time_after;
  // I want to find the number of episodes after which an objective is
  // within a certain percetange of its maximum value


  // Configure IO, ask for user input, and output config files
  // std::cout << "Experiment configs file: " << std::endl;
  // std::cout << config << std::endl;

  YAML::Node config = YAML::LoadFile("../input/config.yaml");

  int trialNum = 1;//readTrialNum();
  string fileDir = "Results/" + std::to_string(trialNum);
  makeDir(fileDir);

  vector< Objective* > objs;
  G global(3,4,1);
  objs.push_back(&global);
  
  YAML::Node experiments = nodeFromYAML(config, "experiments");
  vector<std::string> experimentStrings;

  for (YAML::const_iterator it = experiments.begin(); it != experiments.end(); ++it) {
    std::string experimentName = it->as<std::string>();
    experimentStrings.push_back(experimentName);
  }

  vector< Env* > envs;
  
  vector< vector<size_t>> inds;
  vector< vector< NeuralNet* > > teams;
  
  MultiRover* domain;
  Objective* o;

  VectorXd input;
  for (auto& expKey : experimentStrings) {
    YAML::Node expNode = nodeFromYAML(config, expKey);
    o = objFromYAML(expNode, objectiveS);

    // Begin section to find best agent team
    vector< NeuralNet* > bestTeam;

    vector<double> scores;
    double bestScore = 0.0;

    // Train a network team on the sub-objective. The best representative
    //   team from 10 pools (as based on 25 test runs) will be chosen.
    std::cout << "Training neural network team for " << expKey;
    for (size_t t = 0; t < 1; t++) {
      std::cout << ".";
      std::flush(std::cout);
      domain = getDomain(expNode);
      vector<size_t> agents(domain->getNRovers(), 0);
      trainDomain(domain, expNode, expKey, fileDir, o);
      for (size_t reps = 0; reps < 1; reps++) {
      	domain->InitialiseEpoch();
      	domain->ResetEpochEvals();
      	scores.push_back(domain->runSim(domain->createSim(domain->getNPop()), agents, o));
      }
      
      double newScore = mean(scores);
      if (newScore >= bestScore) {
	bestScore = newScore;
	bestTeam = domain->getNNTeam();
      }
      
    }
    std::cout << " Done." << std::endl;
    teams.push_back(bestTeam);
    for (size_t test = 0; test < 100; test++) {
      input.setZero(4,1);
      input(0) = 1.24;
      input(1) = 0.39;
      input(2) = test * 0.02;
      input(3) = 0.13;
      std::cout << "A3 is: " << input(2) << std::endl;
      inspectAgent(domain, input);
    }
    vector<size_t> ind = fromYAML<vector<size_t>>(expNode, "ind");
    inds.push_back(ind);
    objs.push_back(o);
  } // for each experiment
  
   //input.setZero(4,1);
  //input(0) = 2;
  //    input(4) = 1;
  
  //inspectAgent(domain, input);

  // o = &global;
  // YAML::Node root = nodeFromYAML(config, "control");

  // // Control World
  // // std::cout << "Training control";
  // // vector<double> con_scores;
  // // for (size_t t = 0; t < 50; t++) {
  // //   std::cout << ".";
  // //   std::flush(std::cout);
  // //   domain = getDomain(root);
  // //   vector<size_t> agents(domain=->getNRovers(), 0);
  // //   trainDomain(domain, root, "control", fileDir, o);
  // //   for (size_t reps = 0; reps < 50; reps++) {
  // //     domain->InitialiseEpoch();
  // //     domain->ResetEpochEvals();
  // //     con_scores.push_back(domain->runSim(domain->createSim(domain->getNPop()), agents, o));
  // //   }
  // // }
  // // std::cout << std::endl;
  // // std::cout << "Control Mean Score: " << mean(con_scores) << " StdDev: " << stddev(con_scores) << std::endl;
  
  // Alignments as(objs, 10);
  

  // // Alignment Domain
  // // Get variables from node to construct domain
  // size_t nRovs  = size_tFromYAML(root, nRovsS);
  // size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  // size_t nSteps = size_tFromYAML(root, nStepsS);

  // string type = stringFromYAML(root, typeS);

  // double xmin = fromYAML<double>(root, xminS);
  // double ymin = fromYAML<double>(root, yminS);
  // double xmax = fromYAML<double>(root, xmaxS);
  // double ymax = fromYAML<double>(root, ymaxS);
  // std::vector<double> world = {xmin, xmax, ymin, ymax};

  // vector< Agent* > roverTeam;
  // for (size_t i = 0; i < nRovs; i++) {
  //   vector<NeuralNet*> netsAgent;
  //   for (size_t j = 0; j < teams.size(); j++) {
  //     int netI = teams[j].size() > i ? i : 0;
  //     netsAgent.push_back(teams[j][netI]);
  //   }

  //   roverTeam.push_back(new AlignmentAgent(netsAgent, &as, inds));
  // }
  
  // std::cout << "Creating domain with alignment agents." << std::endl;
  // MultiRover domainD(world, nSteps, nPOIs, nRovs, roverTeam);

  // int biasStart = intFromYAML(root, biasStartS);
  // if (biasStart == 0) {
  //   domainD.setBias(false);
  // }

  // as.addAlignments(300);
  
  // vector< size_t > agentsForSim(nRovs, 0);
  // for (int m = 1; m < 2; m++) {
  //   //std::cout << "Adding alignment values to map... " << m*50 << std::endl;
  //   //as.addAlignments(50);
  //   domainD.setVerbose(false);
  //   std::vector<double> alignScores;

  //   roverTeam[0]->setOutputBool(true);
  //   for (int reps = 0; reps < 10000; reps++) {
  //     std::cout << "New Epoch..." << std::endl;
  //     domainD.InitialiseEpoch();
  //     alignScores.push_back(domainD.runSim(domainD.createSim(domainD.getNPop()),
  // 					   agentsForSim, o));
  //     domainD.ResetEpochEvals();
  //   }

  //   std::cout << "Alignment size: " << m*50 << " Mean: " << mean(alignScores) << " StdDev: "
  // 	      << stddev(alignScores) << " StdErr: " << statstderr(alignScores) << std::endl;
  // }
  return 0 ;
}
