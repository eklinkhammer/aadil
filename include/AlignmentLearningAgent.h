/*******************************************************************************
AlignmentLearningAgent.h

Rover that has as input the full state space (4 agent quads, 4 poi quads), the
same reward structure as a normal rover.

It uses alignment to train agents over time (gradually lowering the percent
of the time alignment is used).

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

#ifndef ALIGNMENT_LEARNING_AGENT_H
#define ALIGNMENT_LEARNING_AGENT_H

#include "AlignmentAgent.h"
#include "Agents/Agent.h"

#include <cstdlib>
#include <iostream>
#include <ctime>

class AlignmentLearningAgent : public AlignmentAgent {
 public:
 AlignmentLearningAgent(vector<NeuralNet*> ns, Alignments* as,
			vector<vector<size_t>> inds, size_t nPop,
			double r, double lr)
   : AlignmentAgent(ns, as, inds, nPop), currentLearning(r), learningRate(lr) {};

  inline virtual State getNextState(size_t i, vector<State> jointState) const;
  
  virtual void updateAgent() { currentLearning *= learningRate; };
 private:
  double currentLearning;
  double learningRate;
};

#endif // ALIGNMENT_LEARNING_AGENT_H
