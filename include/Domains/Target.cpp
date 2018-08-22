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
    curTime(-1), coupling(couple), currentScore(0) {
  
  nearestObsVector.reserve(coupling);
  
  resetNearestObs();
}

void Target::reset() {
  resetObs();
  currentScore = 0;
  observed = false;
}

void Target::resetObs() {
  std::priority_queue<double, std::vector<double>, std::greater<double>> empty;
  std::swap(obs, empty);
}

void Target::addObservation(Vector2d xy) {
  Vector2d diff = xy - GetLocation();
  double d = diff.norm();

  if (d > obsRadius) { return; }

  obs.push(d);
}

void Target::updateScore() {
  if (obs.size() < (size_t) coupling) { return; }

  observed = true;
  
  double sumObs = 0.0;
  for (int i = 0; i < coupling; i++) {
    sumObs += obs.top();
    obs.pop();
  }

  double meanObsRadius = sumObs / ((double) coupling);
  double score = val / (meanObsRadius < 1 ? 1 : meanObsRadius);

  nearestObs = nearestObs > meanObsRadius ? meanObsRadius : nearestObs;
  currentScore = score > currentScore ? score : currentScore;
}

double Target::getScore(bool update) {
  if (update) {
    updateScore();
  }

  return currentScore;
}

void Target::ObserveTarget(Vector2d xy) {
  Vector2d diff = xy - GetLocation();
  double d = diff.norm();

  if (d > obsRadius) { return; }
  
  int furthestValidObsIndex = -1;
  double furthestValidObs = -1;
  for (size_t i = 0; i < nearestObsVector.size(); i++) {
    if (furthestValidObs <= nearestObsVector[i]) {
      furthestValidObs = nearestObsVector[i];
      furthestValidObsIndex = i;
    }
  }


  if (d > furthestValidObs) { return; }
  //  std::cout << "d: " << d << std::endl;
  //std::cout << "furthestValidObs: " << furthestValidObs << std::endl;
  //std::cout << furthestValidObsIndex << std::endl;
  nearestObsVector[furthestValidObsIndex] = d;

  int numberValid = 0;
  for (size_t i = 0; i < nearestObsVector.size(); i++) {
    if (nearestObsVector[i] <= obsRadius) {
      numberValid++;
    }
  }

  //std::cout << "Number Valid: " << numberValid << std::endl;
  //std::cout << coupling << std::endl;
  if (numberValid != coupling) { return; }
  
  double meanObs = std::accumulate(nearestObsVector.begin(),
  				   nearestObsVector.end(),
  				   0.0) / nearestObsVector.size();

  nearestObs = meanObs < nearestObs ? meanObs : nearestObs;
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
  for (const auto& xy : agentLocs) {
    ObserveTarget(xy);
  }
}

void Target::observeTarget(std::vector<State> jointState) {
  for (const auto& s : jointState) {
    ObserveTarget(s.pos());
  }
}

double Target::GetNearestObs() const {
  return nearestObs;
}

double Target::rewardAtCoupling(int c) {
  return IsObserved() ? GetValue() / (1 > nearestObs ? 1 : nearestObs) : 0;
}
