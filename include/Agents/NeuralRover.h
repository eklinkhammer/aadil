/*******************************************************************************
NeuralRovers.h

Rover that has as input the full state space (4 agent quads, 4 poi quads), the
same reward structure as a normal rover, but its neural network chooses
between two sub-policies

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

#ifndef NEURAL_ROVER_H
#define NEURAL_ROVER_H

#include "Learning/NeuralNet.h"
#include "Rover.h"

#include <iostream>
#include <string>

using std::string;
using std::vector;

class NeuralRover : public Rover {
 public:
  // Order of list: poi net, agent net
  NeuralRover(size_t n, size_t nPop, Fitness f, vector<NeuralNet> ns);

  // Uses the jointState to determine which of its neural networks to use to
  //  determine the action.
  virtual Vector2d ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState);

  vector<NeuralNet> getNets() { return netsX; }

  void setOutputBool(bool);
  void outputTrajectory(std::string);
  
 private:
  vector<NeuralNet> netsX;
  std::ofstream trajFile;

  bool outputTrajs;
};

#endif // NEURAL_ROVER_H
