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
#include <queue>
#include <unordered_map>

TeamForming::TeamForming() : TeamForming(-1, -1, -1) {}
	 
TeamForming::TeamForming(int c, double observationR, double minR)
  : Objective(), coupling(c), observationRadius(observationR), minRadius(minR) {}

double TeamForming::reward(Env* env) {
  //  std::cout << "Teamforming reward." << std::endl;
  vector< Agent* > agents = env->getAgents();
  double reward = 0.0;
  //std::cout << "Teamforming reward." << std::endl;
  // I could do this with team forming agents, but I want to reduce
  //  the dependence on agent types.
  vector< Target > teamFormingAgents;
  vector< vector< State > > allJointStates = env->getHistoryStates();
  //std::cout << "Teamforming reward." << std::endl;
  double maxR = 0;

  //std::cout << "Teamforming reward." << std::endl;
  vector< State > currentStates = allJointStates[0];
  //std::cout << "Teamforming reward." << std::endl;
  for (size_t i = 0; i < agents.size(); i++) {
    Target t(currentStates[i].pos(), 10, coupling, observationRadius, false);
    teamFormingAgents.push_back(t);
    maxR += 10;
  }

  //std::cout << "Teamforming reward. 2" << std::endl;
  // This scores each agent if it is part of a team
  for (size_t agent = 0; agent < teamFormingAgents.size(); agent++) {
    //std::cout << "First line in for loop" << std::endl;
    //std::cout << "Loop value: " << agent << std::endl;
    //std::cout << "Teamforming reward. " << agent << " / " << teamFormingAgents.size() << std::endl;
    Target currentAgent = teamFormingAgents[agent];
    for (size_t t = 0; t < allJointStates.size(); t++) {
      //std::cout << "Teamforming reward." << std::endl;
      vector< State > currentStepStates = allJointStates[t];
      //std::cout << "Teamforming reward. 2" << std::endl;
      currentAgent.setLocation(currentStepStates[agent].pos());
      //std::cout << "Teamforming reward." << std::endl;
      //Vector2d currentLoc = currentAgent.GetLocation();
      for (size_t other = 0; other < agents.size(); other++) {
  	if (other != agent) {
  	  Vector2d otherLoc = currentStepStates[other].pos();
  	  currentAgent.ObserveTarget(otherLoc, t);
  	}
      }
      //std::cout << "Teamforming reward loop end." << std::endl;
    }

    //std::cout << "After loop." << std::endl;
    //double currentR = currentAgent.rewardAtCoupling(coupling);
    // std::cout << currentR << std::endl;
    reward += currentAgent.rewardAtCoupling(coupling);
    //std::cout << "Teamforming reward team loop end." << std::endl;
  }

  
  //std::cout << "Teamforming reward. 2" << std::endl;
  return reward / maxR;
}
