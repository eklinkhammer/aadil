#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"
#include "yaml_constants.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  vector<double> world ;
  world.push_back(WORLD_XMIN); 
  world.push_back(WORLD_XMAX);
  world.push_back(WORLD_YMIN); 
  world.push_back(WORLD_YMAX);

  std::cout << config << std::endl;
  
  int trialNum ;
  std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: " ;
  std::cin >> trialNum ;
  
  MultiRover trainDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs, coupling) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"Results/Multirover_experts/%d",trialNum) ;
  char mkdir[buffSize] ;
  sprintf(mkdir,"mkdir -p %s",fileDir) ;
  system(mkdir) ;
  
  std::cout << "\nWriting log files to: " << fileDir << "\n\n" ;
  
  char eFile[buffSize] ;
  sprintf(eFile,"%s/results.txt",fileDir) ;
  char tFile[buffSize] ;
  sprintf(tFile,"%s/trajectories.txt",fileDir) ;
  char pFile[buffSize] ;
  sprintf(pFile,"%s/POIs.txt",fileDir) ;
  
  char cFile[buffSize] ;
  sprintf(cFile,"%s/config.yaml",fileDir) ;
  std::stringstream fileName ;
  fileName << cFile ;
  std::ofstream configFile ;
  if (configFile.is_open())
    configFile.close() ;
  configFile.open(fileName.str().c_str(),std::ios::app) ;
  configFile << config << std::endl; 
  configFile.close() ;
  
  trainDomain.OutputPerformance(eFile) ;
  
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
      trainDomain.OutputTrajectories(tFile, pFile) ;
    
    trainDomain.ResetEpochEvals() ;
    trainDomain.SimulateEpoch() ;
  }
  
  char NNFile[buffSize] ;
  sprintf(NNFile,"%s/NNs.txt",fileDir) ;
  
  std::cout << "\nWriting final control policies to file...\n" ;
  
  trainDomain.OutputControlPolicies(NNFile) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
