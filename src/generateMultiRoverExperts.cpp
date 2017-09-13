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

void trainDomainF(MultiRover* domain, size_t epochs, bool output,
		 int outputPeriod, string exp) {
  for (size_t n = 0; n < epochs; n++) {
    if (output && (n % outputPeriod == 0 || n == epochs - 1)) {
      std::cout << "Training " << exp << " Episode " << n << "...";
      domain->setVerbose(true);
    } else {
      domain->setVerbose(false);
    }
    domain->EvolvePolicies((n==0));
    domain->InitialiseEpoch();

    domain->ResetEpochEvals();
    domain->SimulateEpoch();
  }
}
std::vector<NeuralNet> trainSimple(vector<double> w, AgentType a, size_t
				   roverCount, size_t poiCount, int
				   couplingReq, size_t nSteps, size_t
				   epochs, bool output, int outputPeriod) {
  string exp;
  if (a == AgentType::A) {
    exp = "T";
  } else if (a == AgentType::P) {
    exp = "P";
  } else if (a == AgentType::E) {
    exp = "E";
  } else {
    exp = "R";
  }
  MultiRover domain(w, nSteps, CCEA_POP, poiCount, Fitness::G, roverCount, couplingReq, a);

  trainDomainF(&domain, epochs, output, outputPeriod, exp);

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
  string trajChoiceFile = fileDir + "/trajectory_choice.txt";
  
  makeDir(fileDir);
  printConfig(configFile);

  // Initialize training domain
  vector<double> world;
  world.push_back(WORLD_XMIN); 
  world.push_back(WORLD_XMAX);
  world.push_back(WORLD_YMIN); 
  world.push_back(WORLD_YMAX);
  
  std::cout << "Pre training ..." << std::endl;
  vector<NeuralNet> aNets = trainSimple(world, AgentType::A, TEAM_ROVERS,
					TEAM_POIS, TEAM_COUPLING, TEAM_STEPS,
					TEAM_EPS, true, 20);
  vector<NeuralNet> pNets = trainSimple(world, AgentType::P, POI_ROVERS,
					POI_POIS, POI_COUPLING, POI_STEPS,
					POI_EPS, true, 20);

  vector<NeuralNet> eNets = trainSimple(world, AgentType::E, EXPLORING_ROVERS,
					EXPLORING_POIS, EXPLORING_COUPLING, EXPLORING_STEPS,
					EXPLORING_EPS, true, 20);
  vector<size_t> aInds = {0, 1, 2, 3};
  vector<size_t> pInds = {4, 5, 6, 7};
  vector<size_t> eInds = {0, 1, 2, 3};
  vector< vector<size_t> > inds = {aInds, pInds, eInds};
  vector< vector<NeuralNet> > nets = {aNets, pNets, eNets};
  
  std::cout << std::endl << "Pre training complete. Now training net of nets: " << std::endl;

  MultiRover trainDomain(world, NEURAL_STEPS, CCEA_POP, NEURAL_POIS, Fitness::G,
			 NEURAL_ROVERS, NEURAL_COUPLING, nets, inds);

  trainDomainF(&trainDomain, NEURAL_EPS, true, 20, "N");
  
  // Experiment starts here

  std::cout << "Test starting..." << std::endl;
  std::cout << std::endl << "Writing log files to: " << fileDir << std::endl
	    << std::endl;
  
  trainDomain.OutputPerformance(resultFile);
  trainDomain.OutputTrajectories(trajFile, poiFile, trajChoiceFile);
  trainDomain.InitialiseEpoch();
  trainDomain.SimulateEpoch();

  std::cout << "Test complete." << std::endl;
  
  
  // //vector<Target> targets = targetsFromYAML(config["targets"].as<YAML::Node>());
  // //vector<Vector2d> xys = vector2dsFromYAML(config["agents"].as<YAML::Node>());
  // int staticOrRandom = 1;
  
  // for (size_t n = 0; n < NEURAL_EPS; n++){
  //   std::cout << "Episode " << n << "...";
  //   if (n == 0) {
  //     trainDomain.EvolvePolicies(true) ;

  //     if (staticOrRandom == 0) {
  //       // trainDomain.InitialiseEpochFromVectors(targets, xys) ; // Static world
  //     }
  //   }
  //   else
  //     trainDomain.EvolvePolicies() ;

  //   if (staticOrRandom == 1) {
  //     trainDomain.InitialiseEpoch() ; // Random worlds
  //   }
    
  //   if (n == NEURAL_EPS-1) {
  //     trainDomain.OutputTrajectories(trajFile, poiFile, trajChoiceFile) ;
  //   }
    
  //   trainDomain.ResetEpochEvals() ;
  //   trainDomain.SimulateEpoch() ;
  // }
  

  
  // std::cout << "\nWriting final control policies to file...\n" ;
  
  // trainDomain.OutputControlPolicies(nnFile) ;
  
  // std::cout << "Test complete!\n" ;
  
  return 0 ;
}
