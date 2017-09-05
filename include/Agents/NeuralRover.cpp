/*******************************************************************************
NeuralRovers.cpp

See header file for documentation.

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

#include "NeuralRover.h"

NeuralRover::NeuralRover(size_t n, size_t nPop, Fitness f, vector<NeuralNet> ns)
  : Rover(n, nPop, f), netsX(ns) {}

Vector2d NeuralRover::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState) {
  VectorXd inp = ComputeNNInput(jointState);
  VectorXd out = GetNEPopulation()->GetNNIndex(i)->EvaluateNN(inp).normalized();
  
  VectorXd newInp;
  newInp.setZero(4,1);

  std::cout << "NeuralRover executing control policy." << std::endl;
  if (out(1) > out(0)) {
    newInp(0) = inp(4);
    newInp(1) = inp(5);
    newInp(2) = inp(6);
    newInp(3) = inp(7);
    out = netsX[0].EvaluateNN(newInp).normalized();
  } else {
    newInp(0) = inp(0);
    newInp(1) = inp(1);
    newInp(2) = inp(2);
    newInp(3) = inp(3);
    out = netsX[1].EvaluateNN(newInp).normalized();
  }

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
