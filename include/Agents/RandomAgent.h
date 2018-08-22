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

#ifndef RANDOM_AGENT_H_
#define RANDOM_AGENT_H_

#include "Agent.h"
#include <random>

class RandomAgent : public Agent {
 public:
 RandomAgent() : Agent(1, 1, 1, 1, 1, Fitness::G) {};
  
  virtual State getNextState(size_t, vector<State>) const;
  virtual State getNextState(size_t, VectorXd) const;
  State getNextState() const;
  virtual VectorXd ComputeNNInput(vector<Vector2d> jointState) const;
  virtual void DifferenceEvaluationFunction(vector<Vector2d>, double) {};
  virtual Agent* copyAgent() const;
};

#endif // AGENT_H_
