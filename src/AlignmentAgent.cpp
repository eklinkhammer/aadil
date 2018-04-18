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
  Alignment alignment = alignmentMap->getAlignmentsNN(key);

  // Alignment has global frame dx and dy, and I don't like rotations

  MatrixXd Global2Body = RotationMatrix(-getCurrentPsi());
  Vector2d globalAlign;
  globalAlign.setZero(2,1);
  globalAlign(0) = alignment.getVecX();
  globalAlign(1) = alignment.getVecY();

  
  VectorXd out = Global2Body * globalAlign;

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
