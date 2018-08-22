/*******************************************************************************
Single poi (reward based on each agent's observation)

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

#include "OnePoi.h"

// Default values support heterogenous POIs
OnePoi::OnePoi() : OnePoi(-1, -1) {}
OnePoi::OnePoi(double observationR, double minR)
  : Objective(), observationRadius(observationR), minRadius(minR) {}

vector<double> OnePoi::rewardV(Env* env) {
  vector< Target > targets = env->getTargets();
  vector< Agent* > agents = env->getAgents();
  vector< double > rewards;

  double reward = 0.0;
  double maxR = 0.0;

  for (size_t agent = 0; agent < agents.size(); agent++) {
    for (auto& target : targets) {
      target.setCoupling(1);
      target.setObservationRadius(observationRadius);

      target.ResetTarget();

      for (const auto& ss : env->getHistoryStates()) {
	target.ObserveTarget(ss[agent].pos());
	target.resetNearestObs();
      }

      maxR += target.GetValue();
      reward += target.rewardAtCoupling(0);
    }

    rewards.push_back(reward / maxR);
  }

  return rewards;
}
