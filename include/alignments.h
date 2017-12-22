/*******************************************************************************
alignments.h

Given objectives, computes alignment values between them.

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

#ifndef ALIGNMENTS_H_
#define ALIGNMENTS_H_

#include "alignment.h"
#include "Domains/Env.h"
#include "Domains/Objective.h"
#include "Domains/MultiRover.h"

#include "Agents/Rover.h"

#include <vector>
#include <random>

#include "ssrc/spatial/kd_tree.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <cassert>

typedef std::array<double, 8> Point;
typedef ssrc::spatial::kd_tree<Point, std::vector<Alignment> > Tree;

class Alignments {
 public:
  Alignments(std::vector< Objective* >, int numberSamples, double b);

  void addAlignments(int);
  void addAlignments();
  void addAlignments(MultiRover* domain);
  void addAlignments(Env* env);

  std::vector< Alignment > getAlignmentsNN(std::vector< double > input);
  
 private:
  std::vector< Objective* > objs;

  int numSamples;
  double biasT;

  Tree tree;
};


#endif // ALIGNMENTS_H_
