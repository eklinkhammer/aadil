/*******************************************************************************
AlignmentAgent.cpp

Rover that has as input the full state space (4 agent quads, 4 poi quads), the
same reward structure as a normal rover, but it uses alignment to choose between
a set of neural nets (policies).

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

#include "AlignmentAgent.h"

State AlignmentAgent::getNextState(size_t i, vector<State> jointState) const {
  return getNextStateWork(i, jointState);
}
State AlignmentAgent::getNextStateWork(size_t i, vector<State> jointState) const {
  std::vector<double> key = getVectorState(jointState);  
  std::vector< Alignment > alignments = alignmentMap->getAlignmentsNN(key);
  if (alignments.size() < 1) {
    return getCurrentState();
  }

  // if (printOutput) {
  //   std::cout << "Beginning output for alignment movement." << std::endl;
  //   std::cout << "Input state (as vector): < ";
  //   for (const auto& i : key) {
  //     std::cout << i << " ";
  //   }
  //   std::cout << ">" << std::endl;

  //   std::cout << "Alignment values for objectives: " << std::endl;
  //   for (const auto& a : alignments) {
  //     std::cout << a;
  //   }
  // }
  
  Alignment max;
  size_t bestAlign = 0;
  for (size_t align_i = 0; align_i < alignments.size(); align_i++) {
    if (alignments[align_i].alignScore() < max.alignScore() ||
	(alignments[align_i].alignScore() == max.alignScore() &&
	 alignments[align_i].alignMag() > max.alignMag())) {
      max = alignments[align_i];
      bestAlign = align_i;
    }
  }

  // for (const auto& i : netsX) {
  //   std::cout << i << std::endl;
  // }

  NeuralNet* policy = netsX[bestAlign];
  std::vector<size_t> inds = index[bestAlign];

  vector<Vector2d> justPos;
  for (const auto& s : jointState) {
    justPos.push_back(s.pos());
  }

  VectorXd inp = ComputeNNInput(justPos);
  VectorXd newInp;
  newInp.setZero(inds.size(),1); // set to size inds

  int index_input = 0;
  for (size_t i : inds) {
    newInp(index_input) = inp(i);
    index_input++;
  }

  VectorXd out = policy->EvaluateNN(newInp).normalized();

  // Transform to global frame
  Matrix2d Body2Global = RotationMatrix(getCurrentPsi());
  Vector2d deltaXY = Body2Global*out;
  double deltaPsi = atan2(out(1),out(0));
  
  // Move
  Vector2d currentXY = getCurrentXY() + deltaXY;
  double currentPsi = getCurrentPsi() + deltaPsi;
  currentPsi = pi_2_pi(currentPsi);

  State s(currentXY, currentPsi);

  // if (printOutput) {
  //   std::cout << "Chosen objective: " << bestAlign << std::endl;
  //   std::cout << "Initial State: " << getCurrentState() << std::endl;
  //   std::cout << "Next State: " << s << std::endl;
  // }
  return s;
}

Agent* AlignmentAgent::copyAgent() const {
  AlignmentAgent* copy = new AlignmentAgent(netsX, alignmentMap, index);
  return copy;
}