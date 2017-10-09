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
#include <math.h>

#include "Agents/Rover.h"
#include "Agents/NeuralRover.h"
#include "Agents/TeamFormingAgent.h"
#include "Agents/ExploringAgent.h"
#include "Agents/OnlyPOIRover.h"
#include "Agents/Controlled.h"

#include "Env.h"
#include "State.h"

using std::string ;
using std::vector ;
using std::shuffle ;
using namespace Eigen ;

enum class AgentType {A, P, R, M, E, C};

class MultiRover{
  public:
 MultiRover(vector<double> w, size_t nSteps, size_t nPop, size_t nPOIs,
	    string evalFunc, size_t rovs, int c = 1)
   : MultiRover(w, nSteps, nPop, nPOIs, Fitness::G, rovs, c, AgentType::R) {};
  
    /* MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t */
    /* 	       numPOIs, Fitness f, size_t rovs, int c, size_t nInput, size_t */
    /* 	       nHidden, size_t nOutput, AgentType); */

    // MultiRover without neural net information
    MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t numPOIs,
	       Fitness f, size_t rovs, int c, AgentType);

    // MultiRover with NeuralRovers
    MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t numPOIs,
	       Fitness f, size_t rovs, int c, vector<vector<NeuralNet>>,
	       vector< vector<size_t> >, bool controlled = false);
    ~MultiRover();


    // Gets each agent's first neural net. Does not get the best net.
    vector<NeuralNet*> getNNTeam();
    void initRovers();
    
    void InitialiseEpoch();

    void InitialiseEpochFromVectors(vector<Target>, vector<Vector2d>);
    
    void SimulateEpoch(bool train = true) ;
    void EvolvePolicies(bool init = false) ;
    void ResetEpochEvals();

    void resetPOIs();
    void resetAgents();
    void resetDomain();


    void OutputPerformance(std::string) ;
    void OutputTrajectories(std::string, std::string, std::string) ;
    void OutputControlPolicies(std::string) ;

    // read in control policies and execute in random world, store
    //   trajectory and POI results in second and third inputs, team performance
    //   stored in fourth input, fifth-seventh inputs define NN structure
    void ExecutePolicies(string readFile, string storeTraj, string storePOI,
			 string storeEval, size_t numIn, size_t numOut, size_t numHidden);

    // Read NN from file and create rover team with those agents
    void loadNNs(string nnFile, size_t numIn, size_t numHidden, size_t numOut);

    // Read NNs from file and create Neural Rover with those nets as inputs
    void loadNNsNeuralRover(vector<string>, vector<size_t>, vector<size_t>,
			    vector<size_t>, vector<vector<size_t>>);
    
    void setNSteps(size_t n)        { nSteps = n; }
    void setNPop(size_t n)          { nPop = n; }
    void setNPOIs(size_t n)         { nPOIs = n; }
    void setNRovers(size_t n)       { nRovers = n; }
    void setEvalFunc(string s)      { evaluationFunction = s; }
    void setType(AgentType t)       { type = t; }
    void setFitness(Fitness f)      { fitness = f; }
    void setCoupling(int n)         { coupling = n; }
    void setWorld(vector<double> w) { world = w; }
    void setVerbose(bool toggle)    { verbose = toggle; }
    void setBias(bool bias)         { biasStart = bias; }
    
    size_t         getNSteps()   { return nSteps; }
    size_t         getNPop()     { return nPop; }
    size_t         getNPOIs()    { return nPOIs; }
    size_t         getNRovers()  { return nRovers; }
    string         getEvalFunc() { return evaluationFunction; }
    AgentType      getType()     { return type; }
    Fitness        getFitness()  { return fitness; }
    int            getCoupling() { return coupling; }
    vector<double> getWorld()    { return world; }
    bool           getVerbose()  { return verbose; }
    bool           getBias()     { return biasStart; }

    friend std::ostream& operator<<(std::ostream&, const MultiRover&);
    
  private:
    // Objective Functions
    double objectiveRoverObservePOI(vector<vector<State>>, vector<Target>);
    // double objectiveRoverObservePOI();
    // Other functions
    void printPOIs();
    void printJointState(const vector<Vector2d>);
    void agentObserves(Vector2d xy, int time);

    void toggleAgentOutput(bool);

    
    double calculateG();
    double calculateStepwiseG();
    double calculatePOIValue(const Target& poi);

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
    vector<Target> POIs;
    vector<Target> tempPOIs;
    
    bool gPOIObs ;
    bool verbose;
    bool biasStart;
    
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
    std::ofstream trajChoiceFile;
    
    vector< vector<size_t> > RandomiseTeams(size_t) ;
} ;

#endif // MULTIROVER_H_
