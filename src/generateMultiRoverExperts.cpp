#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"
#include "yaml_constants.h"
#include "experimentalSetup.h"

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

std::vector<NeuralNet> trainSimple(vector<double> w, AgentType a) {
  string exp;
  size_t roverCount, poiCount, epochs;
  int couplingReq;
  
  if (a == AgentType::A) {
    exp = "A";
    roverCount = AGENT_ROVERS;
    poiCount = AGENT_POIS;
    couplingReq = AGENT_COUPLING;
    epochs = AGENT_EPS;
  } else if (a == AgentType::P) {
    exp = "P";
    roverCount = POI_ROVERS;
    poiCount = POI_POIS;
    couplingReq = POI_COUPLING;
    epochs = POI_EPS;
  } else {
    exp = "R";
    roverCount = rovs;
    poiCount = nPOIs;
    couplingReq = coupling;
    epochs = nEps;
  }
  MultiRover domain(w, nSteps, nPop, poiCount, Fitness::G, roverCount, couplingReq, a);

  for (size_t n = 0; n < epochs; n++) {
    std::cout << "Training " << exp << " Episode " << n << "...";
    domain.EvolvePolicies((n==0));
    domain.InitialiseEpoch();

    domain.ResetEpochEvals();
    domain.SimulateEpoch();
  }

  std::vector<NeuralNet*> myNets = domain.getNNTeam();

  std::vector<NeuralNet> returnNets;
  
  for (auto& net : myNets) {
    NeuralNet newNet(net->getNI(), net->getNO(), net->getNH());
    newNet.SetWeights(net->GetWeightsA(), net->GetWeightsB());
    returnNets.push_back(newNet);
  }
  return returnNets;
}

int main() {
  // Configure IO, ask for user input, and output config files
  std::cout << "Experiment configs file: " << std::endl;
  std::cout << config << std::endl;

  int trialNum = readTrialNum();
  string fileDir = "Results/Multirover_experts/" + std::to_string(trialNum);
  string resultFile = fileDir + "/results.txt";
  string trajFile = fileDir + "/trajectories.txt";
  string poiFile = fileDir + "/POIs.txt";
  string configFile = fileDir + "/config.yaml";
  string nnFile = fileDir + "/NNs.txt";
  
  makeDir(fileDir);
  printConfig(configFile);

  // Initialize training domain
  vector<double> world;
  world.push_back(WORLD_XMIN); 
  world.push_back(WORLD_XMAX);
  world.push_back(WORLD_YMIN); 
  world.push_back(WORLD_YMAX);

  //std::cout << "Control..." << std::endl;
  //trainSimple(world, AgentType::R);
  
  std::cout << "Pre training ..." << std::endl;
  vector<NeuralNet> aNets = trainSimple(world, AgentType::A);
  vector<NeuralNet> pNets = trainSimple(world, AgentType::P);

  std::cout << std::endl << "Pre training complete. Now training net of nets: " << std::endl;

  MultiRover trainDomain(world, nSteps, nPop, nPOIs, Fitness::G, rovs, coupling,
			 pNets, aNets);

  // Experiment starts here
  std::cout << std::endl << "Writing log files to: " << fileDir << std::endl
	    << std::endl;
  
  trainDomain.OutputPerformance(resultFile);

  vector<Target> targets = targetsFromYAML(config["targets"].as<YAML::Node>());
  vector<Vector2d> xys = vector2dsFromYAML(config["agents"].as<YAML::Node>());
  
  for (size_t n = 0; n < nEps; n++){
    std::cout << "Episode " << n << "...";
    if (n == 0) {
      trainDomain.EvolvePolicies(true) ;

      if (staticOrRandom == 0) {
        trainDomain.InitialiseEpochFromVectors(targets, xys) ; // Static world
      }
    }
    else
      trainDomain.EvolvePolicies() ;

    if (staticOrRandom == 1) {
      trainDomain.InitialiseEpoch() ; // Random worlds
    }
    
    if (n == nEps-1) {
      trainDomain.OutputTrajectories(trajFile, poiFile) ;
    }
    
    trainDomain.ResetEpochEvals() ;
    trainDomain.SimulateEpoch() ;
  }
  

  
  std::cout << "\nWriting final control policies to file...\n" ;
  
  trainDomain.OutputControlPolicies(nnFile) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
