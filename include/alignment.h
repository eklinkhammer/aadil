/*******************************************************************************
Alignment.h

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

#ifndef alignment_H_
#define alignment_H_

#include <float.h>
#include <vector>
#include <iostream>


/**
   New alignment is defined as a property of not only a point, but a direction.

   1 - strongly aligned
   2 - weakly aligned
   3 - Not anti-aligned
   4 - anti-aligned
 **/
class Alignment {
 public:
 Alignment(int r, int r2) : Alignment( (double) r, (double) r2) {};
 Alignment() : Alignment(1, -DBL_MAX, 0, 0) {}; // lowest possible alignment
 Alignment(double reward1, double reward2) : Alignment(reward1, reward2, 0, 0) {};
 Alignment(double reward1, double reward2, double ux, double uy) : vecx(ux), vecy(uy) {
    if (reward1 > 0) {
      if (reward2 > 0) {
	align_score = 1;
	align_mag = reward2 / reward1;
      } else if (reward2 == 0) {
	align_score = 3;
	align_mag = reward1;
      } else {
	align_score = 5;
	align_mag = reward2 / reward1;
      }
    } else if (reward1 == 0) {
      if (reward2 > 0) {
	align_score = 2;
	align_mag = reward2;
      } else if (reward2 == 0) {
	align_score = 4;
	align_mag = 0;
      } else {
	align_score = 6;
	align_score = reward2;
      }
    } else {
      if (reward2 > 0) {
	align_score = 7;
	align_mag = reward2 / reward1;
      } else if (reward2 == 0) {
	align_score = 8;
	align_mag = reward1;
      } else {
	align_score = 9;
	align_mag = reward2 / reward1;
      }
    }
  }

  int alignScore() const { return align_score; }
  double alignMag() const { return align_mag; }

  double getVecX() const { return vecx; }
  double getVecY() const { return vecy; }

  void setVecX(double ux) { vecx = ux; }
  void setVecY(double uy) { vecx = uy; }
  
  Alignment mappend(Alignment other) {
    if (other.alignScore() < align_score) {
      return other;
    }

    if (other.alignMag() > align_mag) {
      return other;
    }

    return *this;
  }

  // mconcat = foldl mappend mempty
  // mempty is this
  Alignment mconcat(std::vector<Alignment> others) {
    Alignment best = *this;
    for (const auto& other : others) {
      best = best.mappend(other);
    }

    return best;
  }

  friend std::ostream& operator<<(std::ostream &strm, const Alignment &a) {
    strm << a.align_score << " " << a.align_mag << std::endl;
    return strm;
  }
  
 private:
  int align_score;
  double align_mag;

  double vecx;
  double vecy;
};

#endif //alignment_H_
