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

NeuralRover::NeuralRover(size_t n, size_t nPop, Fitness f, vector<NeuralNet> ns, vector<vector<size_t> > indices, size_t nOut)
  : Rover(n, nPop, 8, 16, nOut, f), netsX(ns), index(indices) {}

Vector2d NeuralRover::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState) {
  VectorXd inp = ComputeNNInput(jointState);
  VectorXd out = GetNEPopulation()->GetNNIndex(i)->EvaluateNN(inp).normalized();
  

  int max_index = 0;
  for (int i = 0; i < out.size(); i++) {
    if (out(i) > out(max_index)) {
      max_index = i;
    }
  }

  if (printOutput) {
    outputFile << max_index << std::endl;
  }
  
  vector<size_t> inds = index[max_index];

  VectorXd newInp;
  newInp.setZero(inds.size(),1); // set to size inds
  
  int index_input = 0;
  for (size_t i : inds) {
    newInp(index_input) = inp(i);
    index_input++;
  }

  out = netsX[max_index].EvaluateNN(newInp).normalized();

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

