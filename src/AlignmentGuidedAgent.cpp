/*******************************************************************************
AlignmentGuidedAgent.cpp

See header file for documentation.

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

#include "AlignmentGuidedAgent.h"

size_t AlignmentGuidedAgent::selectIndexOfNetworks(vector<State> jointState) const {
  std::vector<double> key = getVectorState(jointState);
  std::vector<Alignment> alignments = alignmentMap->getAllAlignments(key);
  Alignment best;
  size_t index;
  for (size_t align = 0; align < alignments.size(); align++) {
    Alignment alignment = alignments[align];
    if (best.alignScore() > alignment.alignScore()) {
      index = align;
      best = alignment;
    } else if (best.alignScore() == alignment.alignScore() &&
	       best.alignMag() < alignment.alignMag()) {
      index = align;
      best = alignment;
    }
  }

  //  std::cout << "Index_chosen: " << index << std::endl;

  return index;
}
