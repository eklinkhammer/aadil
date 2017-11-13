/*******************************************************************************
TeamForming.h

Team formation reward. Variation of classic rover domain, where agents count
  as their own pois.

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

#ifndef TeamForming_H_
#define TeamForming_H_

#include "Objective.h"
#include <vector>

using std::vector;

class TeamForming : public Objective {
 public:
  TeamForming();
  TeamForming(int c, double observationR, double minR);
  
  /**
    The classic rover domain reward, as described in <paper>. The reward is the
       sum of the value, per poi, of the closest c agents to the poi scaled by
       their average distance to the poi. Only observations within some radius
       r are counted. No reward is given if less than c agents observe a poi,
       and no additional reward is given for additional agents.

    The coupling (c) value is determined either by the coupling member variable,
      or if set to a default value (-1) will use the target's values.

    The observational radius (r) value is determined by the observationRadius
      member variable, of if set to a default value (less than 0) will use the
      target's observation radius.

    Returns the reward of the current state of the environment.
   **/
  virtual double reward(Env* env);

 private:
  int coupling;
  double observationRadius;
  double minRadius;
};

#endif//TeamForming_H_
