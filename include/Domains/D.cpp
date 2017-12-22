/*******************************************************************************
D.h

Rover domain classic reward, D

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

#include "D.h"


// Default values support heterogenous POIs
D::D() : D(-1, -1, -1) {}
	 
D::D(int c, double observationR, double minR)
  : Objective(), coupling(c), observationRadius(observationR), minRadius(minR) {}

double D::reward(Env* env) {

  G g(coupling, observationRadius, minRadius);

  return g.reward(env);
}

vector<double> D::rewardV(Env* env) {
  vector< Target > targets = env->getTargets(); // copied
  vector< Agent* > agents = env->getAgents();
  vector< double > rewards;
  
  double reward = 0.0;

  G g(coupling, observationRadius, minRadius);
  double gR = g.reward(env);
  
  double maxR = 0.0;
  //std::cout << "Number of agents: " << env->getHistoryStates()[0].size() << std::endl;
  //std::cout << "Number of pois: " << targets.size() << std::endl;
  // std::cout << "G Operator " << targets.size() << std::endl;

  for (size_t agent = 0; agent < agents.size(); agent++) {

    reward = 0;
    maxR = 0;
    
    for (auto& target : targets) {

      if (coupling != -1) {
	target.setCoupling(coupling);
      }

      if (observationRadius != -1) {
	target.setObservationRadius(observationRadius);
      }

      target.ResetTarget();
      
      for (const auto& ss : env->getHistoryStates()) {
	for (size_t a = 0; a < ss.size(); a++) {
	  if (a != agent) {
	    target.ObserveTarget(ss[a].pos());
	  }
	}
	target.resetNearestObs();
      }

      maxR += target.GetValue();
      reward += target.rewardAtCoupling(0);
    }

    rewards.push_back(gR - reward / maxR);
  }

  return rewards;
}
