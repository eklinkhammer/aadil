/*******************************************************************************
NeuralRovers.h

Rover that has as input the full state space (4 agent quads, 4 poi quads), the
same reward structure as a normal rover, but its neural network chooses
between two sub-policies

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

NeuralRover::NeuralRover(size_t n, size_t nPop, Fitness f, NeuralNet* pNet,
			 NeuralNet* aNet)
  : Rover(n, nPop, f), poiNet(pNet), agentNet(aNet) {}

Vector2d NeuralRover::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState) {
  VectorXd inp = ComputeNNInput(jointState);
  VectorXd out = GetNEPopulation()->GetNNIndex(i)->EvaluateNN(inp).normalized();
  
  NeuralNet* net = out(1) > out(0) ? poiNet : agentNet;

  out = net->EvaluateNN(inp).normalized();

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
