/*******************************************************************************
Controlled.cpp

NeuralRover Agent that allows users to manually choose and option. Documentation
in header file.

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

#include "Controlled.h"

Controlled::Controlled(size_t n, size_t nPop, Fitness f, vector<NeuralNet*> ns,
	     vector<vector<size_t>> indices, size_t nOut)
    : NeuralRover(n, nPop, f, ns, indices, nOut) {}

State Controlled::getNextState(size_t i, vector<State> jointState) const {
  vector<Vector2d> options;
  vector<double> psiOptions;

  vector<Vector2d> justPos;
  for (const auto& s : jointState) {
    justPos.push_back(s.pos());
  }

  VectorXd inp = ComputeNNInput(justPos);
  
  for (size_t i = 0; i < netsX.size(); i++) {
    vector<size_t> inds = index[i];

    VectorXd newInp;
    newInp.setZero(inds.size(),1);

    int index_input = 0;

    for (size_t i : inds) {
      inp(index_input) = inp(i);
      index_input++;
    }

    VectorXd out = netsX[i]->EvaluateNN(newInp).normalized();

    // Transform to global frame
    Matrix2d Body2Global = RotationMatrix(getCurrentPsi());
    Vector2d deltaXY = Body2Global*out;
    double deltaPsi = atan2(out(1),out(0));

    options.push_back(deltaXY);
    psiOptions.push_back(deltaPsi);
  }

  for (size_t i = 0; i < options.size(); i++) {
    std::cout << i << ":" << options[i](0) << "," << options[i](1) << " "
	      << psiOptions[i] << std::endl;
  }

  int selection;
  std::cout << "Select option: " << std::endl;
  std::cin >> selection;

  Vector2d deltaXY = options[selection];
  double deltaPsi = psiOptions[selection];

  // Move
  Vector2d currentXY = getCurrentXY() + deltaXY;
  double currentPsi = getCurrentPsi() + deltaPsi;
  currentPsi = pi_2_pi(currentPsi);

  State s(currentXY, currentPsi);
  return s;
}
