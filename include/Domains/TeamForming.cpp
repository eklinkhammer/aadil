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

double TeamForming::operator() (Env* env) {
  vector< Agent* > agents = env->getAgents();
  double reward = 0.0;
  
  for (auto& agent : agents) {
    TeamFormingAgent* t = (TeamFormingAgent*) agent;
    t->setObservationRadius(observationRadius);
    reward += t->rewardAtCoupling(coupling);
  }

  return reward;
}
