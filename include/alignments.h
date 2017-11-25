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

#include <unordered_map> // in place of kdtree (O(nd) nn search)

#include <vector>
#include <random>

struct KeyHash {
  std::size_t operator()(const std::vector<double>& vec) const {
    std::size_t seed = 0;
    for (const auto& v_i : vec) {
      seed ^= std::hash<double>{}(v_i) + 0x9e37779b9 + (seed << 6) + (seed >> 2);
    }
    
    return seed;
  }
};

struct KeyEqual {
  bool operator() (const std::vector<double> & lhs, const std::vector<double> & rhs) const {
    if (lhs.size() != rhs.size()) return false;

    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs[i] != rhs[i]) return false;
    }
    
    return true;
  }
};

class Alignments {
 public:
  Alignments(std::vector< Objective* >, int numberSamples);

  void addAlignments(int);
  void addAlignments();
  void addAlignments(MultiRover* domain);
  void addAlignments(Env* env);

  std::vector< Alignment > getAlignmentsNN(std::vector< double > inputState);
  
 private:
  std::unordered_map< std::vector< double >, std::vector< Alignment >, KeyHash, KeyEqual > alignments;
  std::vector< Objective* > objs;
  
  double distance(std::vector<double>, std::vector<double>);
  int numSamples;
};


#endif // ALIGNMENTS_H_
