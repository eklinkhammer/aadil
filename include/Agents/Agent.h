/*******************************************************************************
Agent.h

Agent class header. Agent is the base class for different types of rovers that
calculate their input vectors differently.

Authors: Eric Klinkhammer

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

#ifndef AGENT_H_
#define AGENT_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <float.h>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Utilities/Utilities.h"
#include "Domains/Target.h"

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

using std::string;
using std::vector;
using std::list;
using std::max;
using easymath::pi_2_pi;

enum class Fitness {G, D};

class Agent {
 public:
  Agent(size_t n, size_t nPop, size_t nInput, size_t nHidden, size_t nOutput,
	Fitness f);

  virtual ~Agent();
  // Computes the input to the neural network using a list of actor joint states
  //
  // The returned VectorXd will be the same size as nInput
  //
  // This function is a pure virtual function.
  virtual VectorXd ComputeNNInput(vector<Vector2d> jointState) = 0;

  // Calculates the new XY position from the ith neural network in the CCEA pool
  //   using the result of ComputeNNInput with the jointstate as input.
  //
  // Updates both currentXY and currentPsi
  //
  // Is implemented. Should only be overriden when the NN does not directly return
  //   a new position (ie, when it chooses between two other networks).
  virtual Vector2d ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState);

  // Sets initial simulation parameters, including rover positions. Clears
  //   evaluation storage vector.
  //
  // Type classes would solve this problem...
  virtual void InitialiseNewLearningEpoch(Vector2d xy, double psi);
  
  // Sets initial simulation parameters, including rover positions. Clears
  //   evaluation storage vector. Intended for use with POIs
  //
  // Base functionality is to ignore the first argument.
  virtual void InitialiseNewLearningEpoch(vector<Target>, Vector2d xy, double psi);

  virtual void DifferenceEvaluationFunction(vector<Vector2d>, double) = 0;

  // Returns a jointstate that has the agent's state replaced with a
  //   counterfactual state. The default is the intial state (ie, if the
  //   agent had not moved at all).
  vector<Vector2d> substituteCounterfactual(vector<Vector2d> jointState);
  vector<Vector2d> substituteCounterfactual(vector<Vector2d>, double, double);

  // Evolves the NeuroEvo, using the epochEvals score. When init is true,
  //  only mutates the population.
  void EvolvePolicies(bool init = false);

  // Writes the neural networks to file
  void OutputNNs(std::string);

  // Sets the epoch values to 0 (for new scenario)
  void ResetEpochEvals();

  // Sets the performance for the epoch and position i to either G or stepwise D
  void SetEpochPerformance(double G, size_t i);

  vector<double> GetEpochEvals() const{ return epochEvals; }
  
  double getCurrentPsi() const { return currentPsi; }
  double getInitialPsi() const { return initialPsi; }

  Vector2d getCurrentXY() const { return currentXY; }
  Vector2d getInitialXY() const { return initialXY; }

  NeuroEvo * GetNEPopulation() const { return AgentNE; }
  
 private:
  size_t nSteps;
  size_t popSize;

  size_t numHidden;
  size_t numOut;

  NeuroEvo* AgentNE;

  std::vector<double> epochEvals;

  void ResetStepwiseEval();


 protected:
  Matrix2d RotationMatrix(double psi);
  Fitness fitness;

  Vector2d initialXY;
  double initialPsi;

  Vector2d currentXY;
  double currentPsi;

  size_t numIn;
  double stepwiseD;

  // Determines the index of this agent's position in a jointState vector
  size_t selfIndex(vector<Vector2d> jointState);
};

#endif // AGENT_H_
