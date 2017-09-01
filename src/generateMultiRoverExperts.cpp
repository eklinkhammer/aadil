#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"
#include "yaml_constants.h"

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

std::string getFileDir(int trialNum) {
  std::stringstream stream;
  stream << "Results/Multirover_experts/" << trialNum;
  return stream.str();
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

int main(){
  // Configure IO, ask for user input, and output config files
  std::cout << "Experiment configs file: " << std::endl;
  std::cout << config << std::endl;

  int trialNum = readTrialNum();
  std::string fileDir = getFileDir(trialNum);
  std::string resultFile = fileDir + "/results.txt";
  std::string trajFile = fileDir + "/trajectories.txt";
  std::string poiFile = fileDir + "/POIs.txt";
  std::string configFile = fileDir + "/config.yaml";
  std::string nnFile = fileDir + "/NNs.txt";
  
  makeDir(fileDir);
  printConfig(configFile);

  // Initialize training domain
  vector<double> world ;
  world.push_back(WORLD_XMIN); 
  world.push_back(WORLD_XMAX);
  world.push_back(WORLD_YMIN); 
  world.push_back(WORLD_YMAX);
  MultiRover trainDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs, coupling);

  // Experiment starts here
  std::cout << std::endl << "Writing log files to: " << fileDir << std::endl
	    << std::endl;
  trainDomain.OutputPerformance(resultFile);
  
  for (size_t n = 0; n < nEps; n++){
    std::cout << "Episode " << n << "..." ;
    if (n == 0){
      trainDomain.EvolvePolicies(true) ;
      if (staticOrRandom == 0)
        trainDomain.InitialiseEpoch() ; // Static world
    }
    else
      trainDomain.EvolvePolicies() ;
    
    if (staticOrRandom == 1)
      trainDomain.InitialiseEpoch() ; // Random worlds
    
    if (n == nEps-1)
      trainDomain.OutputTrajectories(trajFile, poiFile) ;
    
    trainDomain.ResetEpochEvals() ;
    trainDomain.SimulateEpoch() ;
  }
  

  
  std::cout << "\nWriting final control policies to file...\n" ;
  
  trainDomain.OutputControlPolicies(nnFile) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
