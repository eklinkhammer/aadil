/*******************************************************************************
Objective.h

Receives an award from an environment (memoryless)

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

#ifndef OBJ_H_
#define OBJ_H_

#include "Env.h"
#include <float.h>
#include <vector>
#include <string>

class Objective {
public:

  /**
     Reward and rewardV are overloaded for difference reward and regular objectives.

     When there is one reward for the team (that will be passed to all agents), reward 
       returns that reward and rewardV returns that reward duplicated.

     When there is a reward per agent, reward returns the max * mean of rewardV, and rewardV
       returns the per agent reward.
   **/
  virtual double reward(Env* e) {
    std::vector<double> rewards = rewardV(e);

    double sum = 0, maxR = DBL_MIN;
    for (const auto& r : rewards) {
      sum += r;
      if (r > maxR) {
	maxR = r;
      }
    }

    double mean = sum / ((double) rewards.size());
    return maxR * mean;
  }
  
  virtual std::vector<double> rewardV(Env* e) {
    std::vector<double> rewards(e->getAgents().size(), reward(e));
    return rewards;
  }
  
  virtual std::string getName() { return "Objective"; }
};
#endif//OBJ_H_
