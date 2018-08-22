#ifndef NEUR0_EVO_H_
#define NEUR0_EVO_H_

#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include "NeuralNet.h"
#include <string>

using std::vector ;
using std::sort ;
using namespace Eigen ;

class NeuroEvo{
 public:
 NeuroEvo(size_t nIn, size_t nOut, size_t nHidden, size_t pSize)
   : NeuroEvo(nIn, nOut, nHidden, pSize, "BinaryTournament", 0.5) {};
  
  NeuroEvo(size_t nIn, size_t nOut, size_t nHidden, size_t pSize,
	   std::string survivalFunction, double percElite);
  ~NeuroEvo() ;
    
    void MutatePopulation() ;
    void EvolvePopulation(vector<double>) ;
    vector<double> GetAllEvaluations() ;
    
    NeuralNet * GetNNIndex(size_t i){return populationNN[i] ;}
    size_t GetCurrentPopSize() {
      std::cout << "NeuroEvo.h::GetCurrentPopSize()" << std::endl;
      return populationNN.size();
    }
  private:
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    
    size_t populationSize ;
    vector<NeuralNet *> populationNN ;
    
    void (NeuroEvo::*SurvivalFunction)() ;
    void BinaryTournament() ;
    // void RankSelection();
    void RetainBestHalf() ;
    static bool CompareEvaluations(NeuralNet *, NeuralNet *) ;

    double percentElite;
} ;
#endif // NEUR0_EVO_H_
