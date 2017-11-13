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

AlignmentAgent::AlignmentAgent(Fitness f, std::vector<NeuralNet> ns, Alignment* as,
			       std::vector< std::vector<size_t>> indices, size_t nOut)
  : NeuralRover(1,0,f,ns,indices,nOut), alignmentMap(as) {}

State AlignmentAgent::getNextState(size_t i, std::vector<State> jointState) const {
  std::vector<double> key = getVectorState(jointState);
  std::vector< Alignment > alignments = alignmentMap->getAlignmentsNN(key);

  if (alignments.size() < 1) {
    return getCurrentState();
  }
  
  Alignment max;
  size_t bestAlign = -1;
  for (size_t align_i = 0; align_i < alignments.size(); align_i++) {
    if (alignments[align_i].alignScore() > max.alignScore() ||
	(alignments[align_i].alignScore() == max.alignScore() &&
	 alignments[align_i].alignMag() > max.alignMag())) {
      max = alignments[align_i];
      bestAlign = align_i;
    }
  }

  NeuralNet* policy = netsX[align_i];
  std::vector< size_t > indices = index[align_i];

    vector<size_t> inds = index[max_index];

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
  return s;
}
