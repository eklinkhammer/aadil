/*******************************************************************************
TeamFormingAgent.h

Agent that only sees other agents. Extension of the Agent base class. Its 
neural networks' inputs are the 4 Agent quadrants from the Rover domain. It 
scores by the inverse distance to other agents (for some degree of agent 
coupling).

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

#ifndef TEAM_FORMING_AGENT_H
#define TEAM_FORMING_AGENT_H

#include "Agent.h"
#include "Domains/Target.h"

#include <vector>

using std::vector;
using std::max;

class TeamFormingAgent : public Agent, public Target {
 public:
  // TeamFormingAgent has a 4 input neural network. It's reward is the
  //   sum of inverse distances to the nearest tSize other agents.
  TeamFormingAgent(size_t n, size_t nPop, Fitness f, int tSize);

  // Mandatory implementations of virtual functions

  // Computes the NN state for just the agent jointstate. Class has no
  //   POI information to ignore.
  virtual VectorXd ComputeNNInput(vector<Vector2d>);

  // Overriden POI class
  virtual Vector2d GetLocation() const { return getCurrentXY(); }
  
  // Calculates the reward giving 1/r where r is the distance between this
  //   agent and the nearest other agent.
  virtual void DifferenceEvaluationFunction(vector<Vector2d>, double);

  virtual Agent* copyAgent() const; 
  virtual double getReward();
 private:
  size_t teamSize;
};

#endif // TEAM_FORMING_AGENT_H
