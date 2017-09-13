/*******************************************************************************
ExploringAgent.h

Agent that only sees other agents. Extension of the TeamFormingAgent class. Its 
neural networks' inputs are the 4 Agent quadrants from the Rover domain. It 
scores by the distance to other agents (for some degree of agent 
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

#ifndef EXPLORING_AGENT_H
#define EXPLORING_AGENT_H

#include "TeamFormingAgent.h"

class ExploringAgent : public TeamFormingAgent {
 public:
  // An exploring agent receives a negative reward when it is in
  //   a team of tSize or more agents.
  ExploringAgent(size_t n, size_t nPop, Fitness f, int tSize);

  virtual double getReward();
};

#endif // EXPLORING_AGENT_H
