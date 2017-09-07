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
				   fitness(f) {
  AgentNE = new NeuroEvo(numIn, numOut, numHidden, nPop);
}

Agent::~Agent() {
  delete(AgentNE);
  AgentNE = 0;
}

Vector2d Agent::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState) {
  VectorXd inp = ComputeNNInput(jointState);
  VectorXd out = AgentNE->GetNNIndex(i)->EvaluateNN(inp).normalized();

  // Transform to global frame
  Matrix2d Body2Global = RotationMatrix(currentPsi);
  Vector2d deltaXY = Body2Global*out;
  double deltaPsi = atan2(out(1),out(0));
  
  // Move
  currentXY += deltaXY;
  currentPsi += deltaPsi;
  currentPsi = pi_2_pi(currentPsi);
  
  return currentXY;
}

void Agent::InitialiseNewLearningEpoch(Vector2d xy, double psi) {
  initialXY.setZero(initialXY.size(),1);
  ResetStepwiseEval();

  initialXY(0) = xy(0);
  initialXY(1) = xy(1);
  initialPsi = psi;

  currentXY = initialXY;
  currentPsi = initialPsi;
}

void Agent::InitialiseNewLearningEpoch(vector<Target> pois, Vector2d xy, double psi) {
  InitialiseNewLearningEpoch(xy, psi);
}

size_t Agent::selfIndex(vector<Vector2d> jointState) {
  size_t ind = -1;
                  
  for (size_t i = 0; i < jointState.size(); i++) {
    if (jointState[i](0) == currentXY(0) && jointState[i](1) == currentXY(1)) {
      return i;
    }
  }

  return ind;
}

vector<Vector2d> Agent::substituteCounterfactual(vector<Vector2d> jointState) {
  return substituteCounterfactual(jointState, initialXY(0), initialXY(1));
}

vector<Vector2d> Agent::substituteCounterfactual(vector<Vector2d> jointState,
						 double x, double y) {
  size_t ind = selfIndex(jointState);
  jointState[ind](0) = x;
  jointState[ind](1) = y;

  return jointState;
}

Matrix2d Agent::RotationMatrix(double psi){
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
