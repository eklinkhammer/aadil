/*******************************************************************************
TeamForming.cpp

Rover domain classic reward.

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

#include "TeamForming.h"
#include <iostream>
TeamForming::TeamForming() : TeamForming(-1, -1, -1) {}
	 
TeamForming::TeamForming(int c, double observationR, double minR)
  : Objective(), coupling(c), observationRadius(observationR), minRadius(minR) {}

double TeamForming::reward(Env* env) {
  vector< Agent* > agents = env->getAgents();
  double reward = 0.0;

  // I could do this with team forming agents, but I want to reduce
  //  the dependence on agent types.
  vector< Target > teamFormingAgents;
  vector< vector< State > > allJointStates = env->getHistoryStates();

  double maxR = 0;
  vector< State > currentStates = allJointStates[0];
  for (size_t i = 0; i < agents.size(); i++) {
    Target t(currentStates[i].pos(), 10, coupling, observationRadius, false);
    teamFormingAgents.push_back(t);
    maxR += 10;
  }

  for (size_t agent = 0; agent < teamFormingAgents.size(); agent++) {
    Target currentAgent = teamFormingAgents[agent];
    for (size_t t = 0; t < allJointStates.size(); t++) {
      vector< State > currentStepStates = allJointStates[t];
      currentAgent.setLocation(currentStepStates[agent].pos());
      Vector2d currentLoc = currentAgent.GetLocation();
      for (size_t other = 0; other < agents.size(); other++) {
	if (other != agent) {
	  Vector2d otherLoc = currentStepStates[other].pos();
	  currentAgent.ObserveTarget(otherLoc, t);
	}
      }
    }

    double currentR = currentAgent.rewardAtCoupling(coupling);
    std::cout << currentR << std::endl;
    reward += currentAgent.rewardAtCoupling(coupling);
  }

  return reward / maxR;
}
