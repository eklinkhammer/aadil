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

#include "TeamObjective.h"
#include <iostream>
#include <queue>
#include <unordered_map>

TeamObjective::TeamObjective() : TeamObjective(-1, -1, -1) {}
	 
TeamObjective::TeamObjective(int c, double observationR, double minR)
  : Objective(), coupling(c), observationRadius(observationR), minRadius(minR) {}

double TeamObjective::reward(Env* env) {
  vector< Agent* > agents = env->getAgents();
  // I could do this with team forming agents, but I want to reduce
  //  the dependence on agent types.
  vector< Target > teamFormingAgents;
  vector< vector< State > > allJointStates = env->getHistoryStates();
  // double maxR = 0;
  // vector< State > currentStates = allJointStates[0];
  // for (size_t i = 0; i < agents.size(); i++) {
  //   Target t(currentStates[i].pos(), 10, coupling, observationRadius, false);
  //   teamFormingAgents.push_back(t);
  //   maxR += 10;
  // }
  int c = coupling;
  double teamScore = 0.0;
  double maxTeamScore = 1.0 * (agents.size() / c);
  for (auto& currentStates : allJointStates) {
    int teamsFormed = 0;
    // Find team leaders by ranking by he sum of the distances to the rest of the team
    vector< double > distanceSum;
    for (size_t agent = 0; agent < currentStates.size(); agent++) {
      std::priority_queue<double, vector<double>, std::greater<double>> q;
      for (size_t other = 0; other < currentStates.size(); other++) {
	if (agent == other) continue;
	q.push(currentStates[agent].distance(currentStates[other]));
      }
      
      if (((int) q.size()) + 1 < c) {
	distanceSum.push_back(observationRadius*c + 1);
      } else {
	double sum = 0.0;
	for (int i = 1; i < c; i++) {
	  sum += q.top();
	  q.pop();
	}
	distanceSum.push_back(sum);
      }
    }

    // Get the top agent team leaders
    std::unordered_map<double, size_t> table;
    std::priority_queue<double, vector<double>, std::greater<double>> q;
    for (size_t i = 0; i < currentStates.size(); i++) {
      q.push(distanceSum[i]);
      table[distanceSum[i]] = i;
    }
    // Now treat each agent in order of its proximity to its team
    vector<bool> agentTaken;
    for (size_t i = 0; i < currentStates.size(); i++) {
      agentTaken.push_back(false);
    }

    while (!q.empty()) {
      size_t posOfCurrent = table[q.top()];
      q.pop();
      if (agentTaken[posOfCurrent]) continue;
      
      // Order agents by distance, and select closest 
      std::priority_queue<double, vector<double>, std::greater<double>> dists;
      std::unordered_map<double, size_t> indexOfDistance;
      for (size_t other = 0; other < currentStates.size(); other++) {
	if (posOfCurrent == other || agentTaken[other]) continue;
	double dist = currentStates[posOfCurrent].distance(currentStates[other]);
	if (dist > observationRadius) continue;
	dists.push(dist);
	indexOfDistance[dist] = other;
      }
      
      if (((int)dists.size()) + 1 < c) continue;
      
      for (int inteam = 1; inteam < c; inteam++) {
	size_t indexOther = indexOfDistance[dists.top()];
	agentTaken[indexOther] = true;
	dists.pop();
      }
      
      agentTaken[posOfCurrent] = true;
      teamsFormed++;
    }

    double tempTeamScore = teamsFormed / maxTeamScore;
    teamScore = teamScore < tempTeamScore ? tempTeamScore : teamScore;
  }

  return teamScore;
  // // This scores each agent if it is part of a team
  // for (size_t agent = 0; agent < teamFormingAgents.size(); agent++) {
  //   Target currentAgent = teamFormingAgents[agent];
  //   for (size_t t = 0; t < allJointStates.size(); t++) {
  //     vector< State > currentStepStates = allJointStates[t];
  //     currentAgent.setLocation(currentStepStates[agent].pos());
  //     Vector2d currentLoc = currentAgent.GetLocation();
  //     for (size_t other = 0; other < agents.size(); other++) {
  // 	if (other != agent) {
  // 	  Vector2d otherLoc = currentStepStates[other].pos();
  // 	  currentAgent.ObserveTarget(otherLoc, t);
  // 	}
  //     }
  //   }

  //   double currentR = currentAgent.rewardAtCoupling(coupling);
  //   // std::cout << currentR << std::endl;
  //   reward += currentAgent.rewardAtCoupling(coupling);
  // }

  // return reward / maxR;
}
