/*******************************************************************************
OnlyPOIRover.h

Rover that only sees POIs. Extension of Rover base class that has 4 inputs to
its neural networks that are the 4 POI quadrants. It is otherwise identical to
the normal rover.

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

#ifndef ONLY_POI_ROVER_H
#define ONLY_POI_ROVER_H

#include "Rover.h"
#include <vector>

using std::vector;
using std::max;


class OnlyPOIRover : public Rover {
 public:

  // OnlyPOIRover has a 4 input neural network. It's reward is identical to
  //   a normal rover (proximity to POIs)
  OnlyPOIRover(size_t n, size_t nPop, Fitness f);
  ~OnlyPOIRover() {};

  // Computes the NN state for just the given POI locations and values in the
  //   world. Ignores all agent information.
  virtual VectorXd ComputeNNInput(vector<Vector2d> jointState) const;

  virtual Agent* copyAgent() const;
};

#endif // ONLY_POI_ROVER_H
