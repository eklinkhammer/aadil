/*******************************************************************************
G.h

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

#include "G.h"

// Default values support heterogenous POIs
G::G() : G(-1, -1, -1) {}
	 
G::G(int c, double observationR, double minR)
  : Objective(), coupling(c), observationRadius(observationR), minRadius(minR) {}

// There are two ways to approach this. The first is the one below, where
//   target's keep track of the best approach at each coupling. The alternative
//   is to use Env's history of states and just apply them, in order, to the poi's
double G::reward(Env* env) {
  vector< Target > targets = env->getTargets(); // copied

  double reward = 0.0;

  //std::cout << "Number of agents: " << env->getHistoryStates()[0].size() << std::endl;
  //std::cout << "Number of pois: " << targets.size() << std::endl;
  // std::cout << "G Operator " << targets.size() << std::endl;
  for (auto& target : targets) {

    target.ResetTarget();
    
    if (coupling != -1) {
      target.setCoupling(coupling);
    }

    if (observationRadius != -1) {
      target.setObservationRadius(observationRadius);
    }

    // std::cout << "G Operator - History of states" << std::endl;
    //std::cout << target.GetLocation() << std::endl;
    for (const auto& ss : env->getHistoryStates()) {
      //      target.observeTarget(ss);
      for (const auto& s : ss) {
	target.ObserveTarget(s.pos());
      }
    }
    
    reward += target.rewardAtCoupling(coupling);
  }

  return reward;
}
