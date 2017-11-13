/*******************************************************************************
Controlled.h

NeuralRover Agent that allows users to manually choose and option

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

#ifndef CONTROLLED_ROVER_H_
#define CONTROLLED_ROVER_H_

#include "NeuralRover.h"

#include <string>
#include <iostream>


using std::string;
using std::vector;

class Controlled : public NeuralRover {
 public:
  Controlled(size_t n, size_t nPop, Fitness f, vector<NeuralNet*> ns,
	       vector<vector<size_t>> indices, size_t nOut);

  // Rover prompts user to select from among the vectors provided
  //   by all control policies.
  virtual State getNextState(size_t i, vector<State> jointState) const;
};
#endif // CONTROLLED_ROVER_H_
