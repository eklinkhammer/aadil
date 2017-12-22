/*******************************************************************************
ExpTarget.cpp

Exponential for each additional POI an agent observes.

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

#include "ExpTarget.h"

ExpTarget::ExpTarget() : ExpTarget(4.0) {}
ExpTarget::ExpTarget(double observationR) : observationRadius(observationR) {}

// double ExpTarget::reward(Env* env) {

//   std::vector<Target> targets = env->getTargets();
//   std::vector<Agent*> agents = env->getAgents();

//   int numT = (int) targets.size();
//   int rewardR = 1 << numT;
//   int reward = 0;
//   std::vector< std::vector<State> > history = env->getHistoryStates();
//   for (size_t agent = 0; agent < agents.size(); agent++) {
//     std::vector<State> agentHistory;
//     int numObserved = 0;
//     int agentReward = 1;
//     for (const auto& ss : history) {
//       agentHistory.push_back(ss[agent]);
//     }

//     for (auto& target : targets) {
//       target.reset();
//       target.setCoupling(1);
//       target.setObservationRadius(observationRadius);

//       for (const auto& s : agentHistory) {
// 	target.addObservation(s);
//       }

//       target.updateScore();
//       if (target.IsObserved()) {
// 	numObserved++;
//       }
//     }

//     agentReward = agentReward << numObserved;
//     agentReward--;
//     double rewardD = (double) agentReward;
    
//     reward += rewardD;
//   }

//   return reward;
// }

std::vector<double> ExpTarget::rewardV(Env* env) {
  std::vector<Target> targets = env->getTargets();
  std::vector<Agent*> agents = env->getAgents();

  std::vector<double> rewards;

  int numT = (int) targets.size();
  int rewardR = 1 << numT;
  double R = (double) rewardR;
  
  std::vector< std::vector<State> > history = env->getHistoryStates();
  for (size_t agent = 0; agent < agents.size(); agent++) {
    std::vector<State> agentHistory;
    int numObserved = 0;
    int agentReward = 1;
    for (const auto& ss : history) {
      agentHistory.push_back(ss[agent]);
    }

    for (auto& target : targets) {
      target.reset();
      target.setCoupling(1);
      target.setObservationRadius(observationRadius);

      for (const auto& s : agentHistory) {
	target.addObservation(s);
      }

      target.updateScore();
      
      if (target.IsObserved()) {
	numObserved++;
      }
    }
				
    agentReward = agentReward << numObserved;
    agentReward--;


    double rewardD = (double) agentReward / R;// / rewardR;
    //std::cout << "AgentReward: " << agentReward << " MaxReward: " << rewardR << " Reward: " << rewardD << std::endl;
    rewards.push_back(rewardD);
  }

  return rewards;
}

