/*******************************************************************************
RandomAgent.h

RandomAgent class header. Random agent takes uniformly random actions.

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

#include "RandomAgent.h"

State RandomAgent::getNextState(size_t, vector<State>) const {
  return getNextState();
}
State RandomAgent::getNextState(size_t, VectorXd) const {
  return getNextState();
}

State RandomAgent::getNextState() const {
  //srand(time(NULL));
  double x,y;
  x = ((double) rand() / (RAND_MAX));
  y = ((double) rand() / (RAND_MAX));
  Vector2d pos(x,y);
  State s(pos,0);
  return s;
}

VectorXd RandomAgent::ComputeNNInput(vector<Vector2d> jointState) const {
  VectorXd blah;
  return blah;
}

Agent* RandomAgent::copyAgent() const {
  RandomAgent* agent = new RandomAgent();
  agent->move(getCurrentState());
  return agent;
}


