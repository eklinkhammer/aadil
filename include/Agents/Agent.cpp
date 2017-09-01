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

VectorXd Agent::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState) {
  VectorXd inp = ComputeNNInput(jointState);
  VectorXd out = AgentNE->getNNIndex(i)->EvaluateNN(inp).normalized();

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
  initalXY.setZero(initialXY.size(),1);
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
