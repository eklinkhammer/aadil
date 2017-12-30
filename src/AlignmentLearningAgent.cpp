/*******************************************************************************
AlignmentLearningAgent.cpp

Rover that has as input the full state space (4 agent quads, 4 poi quads), the
same reward structure as a normal rover.

It uses alignment to train agents over time (gradually lowering the percent
of the time alignment is used).

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

#include "AlignmentLearningAgent.h"

State AlignmentLearningAgent::getNextState(size_t i, vector<State> jointState) const {
  std::srand(std::time(nullptr));
  int random_variable = std::rand();
  double r = (double) random_variable / DBL_MAX;

  if (r < currentLearning) {
    return getNextStateWork(i, jointState);
  } else {
    
    vector<Vector2d> justPos;
    for (const auto& s : jointState) {
      justPos.push_back(s.pos());
    }
    
    VectorXd inp = ComputeNNInput(justPos);
    VectorXd out = GetNEPopulation()->GetNNIndex(i)->EvaluateNN(inp).normalized();
    return getNewCurrent(out);
  }
}
