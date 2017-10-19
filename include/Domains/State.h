/*******************************************************************************
State.h

A state is a tuple of Vector2d position and double orientation. Defined in
header file.

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

#ifndef STATE_H_
#define STATE_H_

#include <vector>
#include <Eigen/Eigen>
#include <iostream>

using std::vector;

class State {
 public:
  State() {
    Vector2d originVec(0,0);
    _pos = originVec;
    _psi = 0;
  }
  
 State(Vector2d position, double angle) : _pos(position), _psi(angle) {};

  Vector2d pos() const { return _pos; }
  double psi() const { return _psi; }

  friend std::ostream& operator<<(std::ostream &strm, const State &state) {
    return strm << "(" << state.pos()(0) << "," << state.pos()(1) << ")," << state.psi();
  }
 private:
  Vector2d _pos;
  double _psi;
};

#endif // STATE_H_
 
