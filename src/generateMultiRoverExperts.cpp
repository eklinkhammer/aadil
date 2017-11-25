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

  for (const auto& agent : agents) {
    vector<State> states = agent->allNextGetNextState(input);
    std::cout << "Current State" << std::endl;
    std::cout << agent->getCurrentState() << std::endl;
    for (const auto& state : states) {
      std::cout << state << std::endl;
    }
  }
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
  G global(4,4,1);
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
    string type = stringFromYAML(expNode, typeS);

    // Section to find best episode length
    for (size_t eps = 100; eps <= 500; eps+=50) {
      vector<double> epScores;
      vector<double> times;
      std::cout << "Training " << expKey << " for " << eps << " episodes ";
      for (size_t repsEp = 0; repsEp < 10; repsEp++) {
    	std::cout << ".";
    	std::flush(std::cout);
    	domain = getDomain(expNode);
    	domain->setVerbose(false);
    	time(&time_before);
    	trainDomain(domain, eps, false, 20, type, fileDir, expKey, true, o);
    	time(&time_after);
    	double diff = difftime(time_after, time_before);
    	times.push_back(diff);

    	vector<size_t> agents(domain->getNRovers(), 0);;
    	vector<double> scores;
    	// Test this trained domain with a random team and get rewards
    	for (size_t reps = 0; reps < 25; reps++) {
    	  domain->InitialiseEpoch();
    	  domain->ResetEpochEvals();
    	  scores.push_back(domain->runSim(domain->createSim(domain->getNPop()), agents, o));
    	} // For test reps

    	epScores.push_back(mean(scores));
      } // Running multiple trials per epsidoe length (for repsEp)
      std::cout << " Mean: " << mean(epScores) << "Stddev: " << stddev(epScores)
		<< " Time: " << mean(times) << std::endl;
    } // For eps
    
    // End section to find best episode length

    // Begin section to find best agent team
    vector< NeuralNet* > bestTeam;
    vector<size_t> agents(domain->getNRovers(), 0);
    vector<double> scores;
    double bestScore = 0.0;

    std::cout << "Training neural network team for " << expKey;
    for (size_t t = 0; t < 10; t++) {
      std::cout << ".";
      std::flush(std::cout);
      domain = getDomain(expNode);
      domain->setVerbose(false);
      trainDomain(domain, expNode, expKey, fileDir, o);
      for (size_t reps = 0; reps < 25; reps++) {
	domain->InitialiseEpoch();
	domain->ResetEpochEvals();
	scores.push_back(domain->runSim(domain->createSim(domain->getNPop()), agents, o));
      }
      
      double newScore = mean(scores);
      if (newScore > bestScore) {
	bestScore = newScore;
	bestTeam = domain->getNNTeam();
      }
      
    }
    std::cout << " Done." << std::endl;
    teams.push_back(bestTeam);
    // End section that finds best agent team
    //trainDomain(domain, expNode, expKey, fileDir, o);
    vector<size_t> ind = fromYAML<vector<size_t>>(expNode, "ind");
    inds.push_back(ind);
    //teams.push_back(domain->getNNTeam());
    objs.push_back(o);
  } // for each experiment
  
   //input.setZero(4,1);
  //input(0) = 2;
  //    input(4) = 1;
  
  //inspectAgent(domain, input);

  // Control World
  std::cout << "Training control..." << std::endl;
  YAML::Node root = nodeFromYAML(config, "NeuralRover");
  domain = getDomain(root);
  trainDomain(domain, root, "Control", fileDir, &global);
  
  Alignments as(objs, 10);
  

  // Alignment Domain
  // Get variables from node to construct domain
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);

  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};

  // bool controlled = false;
  // int controlInt = intFromYAML(root, "input");
  // if (controlInt == 1) {
  //   controlled = true;
  // }

  vector< Agent* > roverTeam;
  for (size_t i = 0; i < nRovs; i++) {
    vector<NeuralNet*> netsAgent;
    for (size_t j = 0; j < teams.size(); j++) {
      int netI = teams[j].size() > i ? i : 0;
      netsAgent.push_back(teams[j][netI]);
    }

    roverTeam.push_back(new AlignmentAgent(netsAgent, &as, inds));
  }
  
  std::cout << "Creating domain with alignment agents." << std::endl;
  MultiRover domainD(world, nSteps, nPOIs, nRovs, roverTeam);

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domainD.setBias(false);
  }

  vector< size_t > agentsForSim(nRovs, 0);
  for (int m = 1; m < 17; m++) {
    std::cout << "Adding alignment values to map... " << m*50 << std::endl;
    as.addAlignments(50);
    domainD.setVerbose(false);
    domain->setVerbose(false);
    std::vector<double> alignScores;
    std::vector<double> policyScores;

    for (int reps = 0; reps < 100; reps++) {
      //std::cout << "Aignment Score: ";
      domainD.InitialiseEpoch();
      domain->InitialiseEpochFromOtherDomain(&domainD);

      
      alignScores.push_back(domainD.runSim(domainD.createSim(domainD.getNPop()),
					   agentsForSim, o));
      policyScores.push_back(domain->runSim(domain->createSim(domain->getNPop()),
					    agentsForSim, o));
      domainD.ResetEpochEvals();
      domain->ResetEpochEvals();
    }

    std::cout << "Alignment Mean: " << mean(alignScores) << " StdDev: "
	      << stddev(alignScores) << " StdErr: " << statstderr(alignScores) << std::endl;
    
    std::cout << "Policy Mean:    " << mean(policyScores) << " StdDev: "
	      << stddev(policyScores) << " StdErr: " << statstderr(policyScores) << std::endl;
  }
  return 0 ;
}
