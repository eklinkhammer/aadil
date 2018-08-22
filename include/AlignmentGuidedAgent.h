/*******************************************************************************
AlignmentGuidedAgent.h

Rover that selects from policies based on alignment values.

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

#ifndef ALIGNMENT_GUIDED_AGENT_H
#define ALIGNMENT_GUIDED_AGENT_H

#include "Agents/NeuralRover.h"
#include "alignments.h"

class AlignmentGuidedAgent : public NeuralRover {
 public:
  AlignmentGuidedAgent(vector<NeuralNet*> ns, Alignments* as,
		       vector<vector<size_t>> inds)
    : NeuralRover(1, 0, Fitness::G, ns, inds, 1), alignmentMap(as) {};

  virtual size_t selectIndexOfNetworks(vector<State> jointState) const;
  
 private:
  Alignments* alignmentMap;
};

#endif // ALIGNMENT_GUIDED_AGENT_H
