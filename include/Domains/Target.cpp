/*******************************************************************************
Target.cpp

Target class source. Targets record observations and maintain a record of the 
closest one, or closest set if multiple observations are required. Comments and
documentation can be found in the header file.

Authors: Eric Klinkhammer, Jen Jen Chung

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

#include <iostream>
#include "Target.h"

Target::Target(Vector2d xy, double value, int couple, double obsR, bool obs)
  : loc(xy), val(value), obsRadius(obsR), nearestObs(DBL_MAX), observed(obs),
    curTime(-1), coupling(couple), maxConsideredCouple(3) {
  
  nearestObsVector.reserve(coupling);
  
  resetNearestObs();

  for (int i = 0; i <= maxConsideredCouple; i++) {
    nearestObservations.push_back(DBL_MAX);
  }
}

void Target::ObserveTarget(Vector2d xy) {
  nearestObs = GetNearestObs();
  Vector2d diff = xy - GetLocation();
  double d = diff.norm();

  if (d > obsRadius) { return; }
  
  int furthestValidObsIndex = -1;
  double furthestValidObs = DBL_MIN;
  for (size_t i = 0; i < nearestObsVector.size(); i++) {
    if (furthestValidObs <= nearestObsVector[i]) {
      furthestValidObs = nearestObsVector[i];
      furthestValidObsIndex = i;
    }
  }

  if (d > furthestValidObs) { return; }
  nearestObsVector[furthestValidObsIndex] = d;

  int numberValid = 0;
  for (size_t i = 0; i < nearestObsVector.size(); i++) {
    if (nearestObsVector[i] <= obsRadius) {
      numberValid++;
    }
  }

  if (numberValid != coupling) { return; }
  
  double meanObs = std::accumulate(nearestObsVector.begin(),
  				   nearestObsVector.end(),
  				   0.0) / nearestObsVector.size();

  nearestObs = meanObs < GetNearestObs() ? meanObs : GetNearestObs();
  observed = true;
}

void Target::ObserveTarget(Vector2d xy, size_t t) {
  if (curTime != t) {
    curTime = t;
    resetNearestObs();
  }

  ObserveTarget(xy);
}

void Target::ResetTarget(){
  nearestObs = DBL_MAX;
  observed = false;
  resetNearestObs();
}

void Target::resetNearestObs() {
  nearestObsVector.clear();
  for (int i = 0; i < coupling; i++) {
    nearestObsVector.push_back(DBL_MAX);
  }
}

void Target::setObservationRadius(double newRadius) {
  if (GetNearestObs() > newRadius) {
    ResetTarget();
  }

  obsRadius = newRadius;
}
std::ostream& operator<<(std::ostream &strm, const Target &t) {
  return strm << t.loc(0) << "," << t.loc(1) << "," << t.val;
}

void Target::observeTargetMultiple(std::vector<Vector2d> agentLocs) {
  for (int i = 1; i <= maxConsideredCouple; i++) {
    setCoupling(i);
    resetNearestObs();
    for (const auto& loc : agentLocs) {
      ObserveTarget(loc);
    }

    nearestObservations[i] = nearestObs;
  }
}

void Target::observeTarget(std::vector<State> jointState) {
  for (const auto& s : jointState) {
    ObserveTarget(s.pos());
  }

  nearestObservations[coupling] = nearestObs;
}

double Target::GetNearestObs() const {
  if (nearestObservations.empty()) {
    return nearestObs;
  } else {
    return nearestObservations[coupling];
  }
}

double Target::rewardAtCoupling(int c) {
  if (nearestObservations[c] > obsRadius) {
    return 0;
  }

  // for (const auto& i : nearestObservations) {
  //   std::cout << i << " ";
  // }
  // std::cout << std::endl;
  if (nearestObservations[c] > DBL_MAX - 1) {
    return 0;
  }
  double reward = GetValue() / (1 > nearestObservations[c] ? 1 : nearestObservations[c]);
  return reward;
}
