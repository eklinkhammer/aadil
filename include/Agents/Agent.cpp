/*******************************************************************************
Agent.cpp

Agent is the base class for different types of rovers that
calculate their input vectors differently. Documentation for all functions
can be found in header file.

Authors: Jen Jen Chung, Eric Klinkhammer

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

#include "Agent.h"

Agent::Agent(size_t n, size_t nPop, size_t nInput, size_t nHidden, size_t
	     nOutput, Fitness f) : nSteps(n), popSize(nPop), numIn(nInput),
				   numHidden(nHidden), numOut(nOutput),
				   fitness(f), printOutput(false) {
  AgentNE = new NeuroEvo(numIn, numOut, numHidden, nPop);

  id = std::chrono::system_clock::now().time_since_epoch().count() % 100000;
}

Agent::~Agent() {
  delete(AgentNE);
  AgentNE = 0;
}

State Agent::executeNNControlPolicy(size_t i, vector<State> jointState) {
  State newState = getNextState(i, jointState);
  move(newState);
  return newState;
}

State Agent::getNextState(size_t i, vector<State> jointState) const {
  vector<Vector2d> justPos;
  for (const auto& s : jointState) {
    justPos.push_back(s.pos());
  }

  VectorXd inp = ComputeNNInput(justPos);
  VectorXd out = AgentNE->GetNNIndex(i)->EvaluateNN(inp).normalized();

  // Transform to global frame
  Matrix2d Body2Global = RotationMatrix(getCurrentPsi());
  Vector2d deltaXY = Body2Global*out;
  double deltaPsi = atan2(out(1),out(0));
  
  // Move
  //std::cout << "Current State: " << getCurrentState();
  //std::cout << " Delta State: " << State(deltaXY, deltaPsi);
  Vector2d currentXY = getCurrentXY() + deltaXY;
  double currentPsi = getCurrentPsi() + deltaPsi;
  currentPsi = pi_2_pi(currentPsi);

  State s(currentXY, currentPsi);
  return s;
}

void Agent::move(State newState) {
  currentState = newState;
}

void Agent::initialiseNewLearningEpoch(State s) {
  initialState = s;
  currentState = initialState;

  ResetStepwiseEval();
}

void Agent::initialiseNewLearningEpoch(State s, vector<Target> targets) {
  initialiseNewLearningEpoch(s);
}

size_t Agent::selfIndex(vector<Vector2d> jointState) const {
  Vector2d currentXY = getCurrentXY();
  for (size_t i = 0; i < jointState.size(); i++) {
    if (jointState[i](0) == currentXY(0) && jointState[i](1) == currentXY(1)) {
      return i;
    }
  }

  return -1;
}

vector<Vector2d> Agent::substituteCounterfactual(vector<Vector2d> jointState) {
  Vector2d initialXY = getInitialXY();
  return substituteCounterfactual(jointState, initialXY(0), initialXY(1));
}

vector<Vector2d> Agent::substituteCounterfactual(vector<Vector2d> jointState,
						 double x, double y) {
  size_t ind = selfIndex(jointState);
  jointState[ind](0) = x;
  jointState[ind](1) = y;

  return jointState;
}

Matrix2d Agent::RotationMatrix(double psi) const{
  Matrix2d R ;
  R(0,0) = cos(psi) ;
  R(0,1) = -sin(psi) ;
  R(1,0) = sin(psi) ;
  R(1,1) = cos(psi) ;
  return R ;
}

void Agent::ResetStepwiseEval(){
  stepwiseD = 0.0 ;
}

void Agent::EvolvePolicies(bool init) {
  //std::cout << "In agent, evolving policy..." << init << std::endl;
  if (!init) {
    //std::cout << "Evolving population..." << std::endl;
    AgentNE->EvolvePopulation(epochEvals);
  }

  // std::cout << "Mutating population..." << std::endl;
  AgentNE->MutatePopulation() ;
}

void Agent::OutputNNs(std::string nnFile) {
  std::ofstream NNFile;
  NNFile.open(nnFile.c_str(),std::ios::app) ;
  
  // Only write in non-mutated (competitive) policies
  for (size_t i = 0; i < popSize; i++){
    NeuralNet * NN = AgentNE->GetNNIndex(i);
    
    MatrixXd NNA = NN->GetWeightsA() ;
    for (int j = 0; j < NNA.rows(); j++){
      for (int k = 0; k < NNA.cols(); k++)
        NNFile << NNA(j,k) << "," ;
      NNFile << std::endl;
    }
    
    MatrixXd NNB = NN->GetWeightsB() ;
    for (int j = 0; j < NNB.rows(); j++){
      for (int k = 0; k < NNB.cols(); k++)
        NNFile << NNB(j,k) << "," ;
      NNFile << std::endl;
    }
  }
  NNFile.close() ;
}

void Agent::ResetEpochEvals(){
  // Re-initialise size of evaluations vector
  vector<double> evals(2*popSize,0) ;
  epochEvals = evals ;
}

void Agent::SetEpochPerformance(double G, size_t i) {
  if (fitness == Fitness::D) {
    epochEvals[i] = stepwiseD;
  } else if (fitness == Fitness::G) {
    epochEvals[i] = G;
  }
}

void Agent::openOutputFile(std::string filename) {
  std::string uniqueFilename = filename + std::to_string(id);
  if (outputFile.is_open()) {
    outputFile.close();
  }

  outputFile.open(uniqueFilename.c_str(), std::ios::app);
}

std::ostream& operator<<(std::ostream &strm, const Agent &a) {
  Vector2d currentXY = a.getCurrentXY();
  return strm << "ID: " << a.id << " Loc: (" << currentXY(0) << ", " << currentXY(1) << ")";
}
