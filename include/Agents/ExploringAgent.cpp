/*******************************************************************************
ExploringAgent.cpp

Agent that only sees other agents. Extension of the TeamFormingAgent class. Its 
neural networks' inputs are the 4 Agent quadrants from the Rover domain. It 
scores by the distance to other agents (for some degree of agent 
coupling). Method documentation in header file.

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

#include "ExploringAgent.h"

ExploringAgent::ExploringAgent(size_t n, size_t nPop, Fitness f, int tSize)
  : TeamFormingAgent(n, nPop, f, tSize) {}

double ExploringAgent::getReward() {
  return IsObserved() ? (GetValue() * (1 - 1 / max(GetNearestObs(), 1.0))) : GetValue();
}

Agent* ExploringAgent::copyAgent() const {
  ExploringAgent* copy = new ExploringAgent(getSteps(), getPop(), getFitness(), getTeamSize());
  copy->setNets(GetNEPopulation());
  return copy;
}
