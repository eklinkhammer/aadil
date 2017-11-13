/*******************************************************************************
AlignmentAgent.h

Rover that has as input the full state space (4 agent quads, 4 poi quads), the
same reward structure as a normal rover, but it uses alignment to choose between
a set of neural nets (policies).

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

#ifndef ALIGNMENT_AGENT_H
#define ALIGNMENT_AGENT_H

#include "Learning/NeuralNet.h"
#include "NeuralRover.h"
#include "alignments.h"

#include <iostream>
#include <string>
#include <vector>
class AlignmentAgent : public NeuralRover {
 public:
  AlignmentAgent(Fitness f, std::vector<NeuralNet*> ns, Alignments* as,
		 std::vector< <std::vector<size_t> > indices, size_t nOut);

  virtual State getNextState(size_t i, std::vector<State> jointState) const;
  
 protected:
  Alignments* alignmentMap;
}

#endif // ALIGNMENT_AGENT_H
