#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"
#include "yaml_constants.h"
#include "experimentalSetup.h"
#include "experimentUtil.h"

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

void printConfig(std::string fileName) {
  std::ofstream stream;
  stream.open(fileName.c_str(), std::ios::app);
  stream << config << std::endl;
  stream.close();
}

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
		 string exp, string topDir, string id) {
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

    trainDomainOnce(domain, (n==0));
  }
}

void trainDomainOnce(MultiRover* domain, bool evolve) {
  domain->EvolvePolicies(evolve);
  domain->InitialiseEpoch();
  domain->ResetEpochEvals();
  domain->SimulateEpoch();
}

void testDomainOnce(MultiRover* domain, bool output) {
  domain->setVerbose(output);
  domain->InitialiseEpoch();
  domain->ResetEpochEvals();
  domain->SimulateEpoch();
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

std::vector<NeuralNet> trainAndGetTeam(YAML::Node root, std::string key, std::string topDir) {
  // Get variables from node to construct domain
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);

  size_t cceaPop = size_tFromYAML(root, cceaPopS);
  size_t nEps = size_tFromYAML(root, nEpsS);


  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};
  
  MultiRover domain(world, nSteps, cceaPop, nPOIs, Fitness::G, nRovs, coupling, t);

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain.setBias(false);
  }

  int output = intFromYAML(root, outputS);
  trainDomain(&domain, nEps, (output == 1), 20, type, topDir, key);

  return getTeam(&domain);
}

Env* trainAndGetEnv(YAML::Node root, std::string key, std::string topDir) {
  // Get variables from node to construct domain
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);

  size_t cceaPop = size_tFromYAML(root, cceaPopS);
  size_t nEps = size_tFromYAML(root, nEpsS);


  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};
  
  MultiRover domain(world, nSteps, cceaPop, nPOIs, Fitness::G, nRovs, coupling, t);

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain.setBias(false);
  }

  int output = intFromYAML(root, outputS);
  trainDomain(&domain, nEps, (output == 1), 20, type, topDir, key);

  vector<Agent*> agents = domain.getAgents();
  Env* env = domain.createSim(nRovs);
  env->setID(key);
  return env;
}

void configureOutput(MultiRover* domain, string fileDir, string id) {
  string resultFile = fileDir + "/" + id + "_results";
  string trajFile   = fileDir + "/" + id + "_trajectory";
  string poiFile    = fileDir + "/" + id + "_POIs";
  string choiceFile = fileDir + "/" + id + "_choices";
  domain->OutputPerformance(resultFile);
  domain->OutputTrajectories(trajFile, poiFile, choiceFile);
}

int main() {
  // Configure IO, ask for user input, and output config files
  std::cout << "Experiment configs file: " << std::endl;
  std::cout << config << std::endl;

  int trialNum = 1;//readTrialNum();
  string fileDir = "Results/" + std::to_string(trialNum);
  makeDir(fileDir);
  
  YAML::Node experiments = nodeFromYAML(config, "experiments");
  vector<std::string> experimentStrings;
  
  for (YAML::const_iterator it = experiments.begin(); it != experiments.end(); ++it) {
    std::string experimentName = it->as<std::string>();
    experimentStrings.push_back(experimentName);
  }

  vector< Env* > envs;
  vector< vector<size_t>> inds;
  
  for (auto& expKey : experimentStrings) {
    YAML::Node expNode = nodeFromYAML(config, expKey);
    
    Env* env = trainAndGetEnv(expNode, expKey, fileDir);
    vector<size_t> ind = fromYAML<vector<size_t>>(expNode, "ind");
    envs.push_back(env);
    inds.push_back(ind);
  }

  YAML::Node root = nodeFromYAML(config, "NeuralRover");

    // Get variables from node to construct domain
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);

  size_t cceaPop = size_tFromYAML(root, cceaPopS);
  size_t nEps = size_tFromYAML(root, nEpsS);


  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};

  bool controlled = false;
  int controlInt = intFromYAML(root, "input");
  if (controlInt == 1) {
    controlled = true;
  }

  std::cout << "Got to domain creation." << std::endl;
  MultiRover domain(world, nSteps, cceaPop, nPOIs, Fitness::G, nRovs, coupling, t);
  //MultiRover domain(world, nSteps, cceaPop, nPOIs, Fitness::G, nRovs, coupling, neuralNets, inds, controlled);

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain.setBias(false);
  }

  domain.setVerbose(true);
  domain.InitialiseEpoch();
  domain.ResetEpochEvals();
  domain.simulateWithAlignment(false, envs);
  return 0 ;
}
