#ifndef MULTIROVER_H_
#define MULTIROVER_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "Agents/Rover.h"
#include "Agents/NeuralRover.h"
#include "Agents/TeamFormingAgent.h"
#include "Agents/OnlyPOIRover.h"

using std::string ;
using std::vector ;
using std::shuffle ;
using namespace Eigen ;

enum class AgentType {A, P, R, M};

class MultiRover{
  public:
 MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t numPOIs,
	    string evalFunc, size_t rovs, int c = 1)
   : MultiRover(w, numSteps, numPop, numPOIs, Fitness::G, rovs, c, 8, 16, 2,
		AgentType::R) {};
  
    MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t
	       numPOIs, Fitness f, size_t rovs, int c, size_t nInput, size_t
	       nHidden, size_t nOutput, AgentType);

    // MultiRover without neural net information
    MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t numPOIs,
	       Fitness f, size_t rovs, int c, AgentType);

    // MultiRover with NeuralRovers
    //
    // Requirement: The size of netsP, netsA, and rovs must be equal
    MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t numPOIs,
	       Fitness f, size_t rovs, int c, vector<NeuralNet>, vector<NeuralNet>);

    // Create a MultiRover Domain from vector of agents and targets
    //MultiRover(vector<double> w, size_t numSteps, size_t numPop,
    //	       vector<Target> target, vector<Agent> agents, Fitness f, int c);
    ~MultiRover();


    // Gets each agent's first neural net. Does not get the best net.
    vector<NeuralNet*> getNNTeam();
    
    void InitialiseEpoch();

    void InitialiseEpochFromVectors(vector<Target>, vector<Vector2d>);
    
    void SimulateEpoch(bool train = true) ;
    void EvolvePolicies(bool init = false) ;
    void ResetEpochEvals() ;
    
    void OutputPerformance(std::string) ;
    void OutputTrajectories(std::string, std::string) ;
    void OutputControlPolicies(std::string) ;
    void OutputQueries(char *) ;
    void OutputBeliefs(char *) ;
    void OutputAverageStepwise(char *) ;
    
    void ExecutePolicies(char * readFile, char * storeTraj, char * storePOI, char * storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in control policies and execute in random world, store trajectory and POI results in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure
    
    void ExecutePolicies(char * expFile, char * novFile, char * storeTraj, char * storePOI, char* storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in expert and novice control policies and execute in random world, store trajectory and POI results in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure
  private:
    Fitness fitness;
    AgentType type;
    
    vector<double> world ;
    size_t nSteps ;
    size_t nPop ;
    size_t nPOIs ;
    string evaluationFunction ;
    size_t nRovers ;
    int coupling ;
    
    vector<Vector2d> initialXYs ;
    vector<double> initialPsis ;
    
    vector<Agent *> roverTeam ;
    vector<Target> POIs ;
    bool gPOIObs ;
    
    bool outputEvals ;
    bool outputTrajs ;
    bool outputNNs ;
    bool outputQury ;
    bool outputBlf ;
    bool outputAvgStepR ;
    
    std::ofstream evalFile ;
    std::ofstream trajFile ;
    std::ofstream POIFile ;
    std::ofstream NNFile ;
    std::ofstream quryFile ;
    std::ofstream blfFile ;
    std::ofstream avgStepRFile ;
    
    vector< vector<size_t> > RandomiseTeams(size_t) ;
} ;

#endif // MULTIROVER_H_
