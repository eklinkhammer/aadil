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
     The reward if agents were treated as POIs.
   **/
  virtual double reward(Env* env);

 private:
  int coupling;
  double observationRadius;
  double minRadius;
};

#endif//TeamForming_H_
