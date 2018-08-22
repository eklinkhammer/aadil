/*******************************************************************************
Target.h

Target class header. Targets record observations and maintain a record of the 
closest one, or closest set if multiple observations are required.

Authors: Jen Jen Chung, Eric Klinkhammer

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

#ifndef TARGET_H_
#define TARGET_H_

#include <float.h>
#include <Eigen/Eigen>
#include <vector>
#include "State.h"
#include <queue>

using namespace Eigen;

class Target{
 public:

  // Default Target Constructors and values:
  //   coupling: 1
  //   observation radius: 4
  //   observed: false
 Target(Vector2d xy, double v) : Target(xy, v, 1) {}
 Target(Vector2d xy, double v, int c) : Target(xy, v, c, 4.0, false) {}

  //Target(

  // Target Constructor
  // xy: The xy location of the Target. ObserveTarget uses this value to
  //        compare distances.
  // value: The score of the Target
  // couple: The number of simultaneous observers required to become observed.
  // obsR: The distance within an observer must be to be counted.
  // obs: Whether the Target has been observed.
  Target(Vector2d xy, double value, int couple, double obsR, bool obs);  

  
  virtual Vector2d GetLocation() const { return loc; }
  double GetValue() const { return val; }
  double GetNearestObs() const;
  bool IsObserved() const { return observed; }

  // The target is observed by an object at position xy. If this observer is
  //   closer than the current set of closest observations (which may be just
  //   of size 1), the internal observation list is updated. The position xy
  //   must be within the observation radius.
  //
  // A Target that has been observed will have its "observed" field set to true,
  //   and this value will not change upon subsequent observations. A target
  //   will only be observed if the sufficient number of unique observers are
  //   present (the number determined by the coupling field).
  void ObserveTarget(Vector2d xy);

  // The target is observed by an object at position xy at time t. For coupling
  //   requirements higher than 1, the observations must be simultaneous. When
  //   given an observation with a time, the target will only consider observers
  //   with that time (or no time) going forward.
  //
  // Targets do not keep records across time, and will not be able to accurately
  //   track out of order observations.
  //
  // For identical times and times that match the current time value, this
  //   function has the same effects as ObserveTarget(Vector).
  void ObserveTarget(Vector2d xy, size_t t);

  void observeTarget(std::vector<State>);
  //void observeTarget(std::vector<State>);
  // Resetting a target sets it to back to being a target that has not yet been
  //   observed by any observer. Its current list of observations is empty. It
  //   has no nearest observation.
  void ResetTarget();
  friend std::ostream& operator<<(std::ostream&, const Target&);

  // The reward of the target at a certain coupling value
  double rewardAtCoupling(int);

  double reward();
  
  void observeTargetMultiple(std::vector<Vector2d>);
  
  void setLocation(Vector2d newLoc) { loc = newLoc; }
  int getCoupling() const { return coupling; }
  double getObservationRadius() const { return obsRadius; }

  // Sets the observation radius of the target. If the old nearestObs was
  //   less than the newRadius, reset.
  void setObservationRadius(double newRadius);
  
  void setCoupling(int newCoupling) {
    coupling = newCoupling;
    if (maxConsideredCouple < coupling) {
      maxConsideredCouple = coupling;
    }
  }

  bool equals(Target t) {
    Vector2d other = t.GetLocation();
    return other(0) == loc(0) && other(1) == loc(1);
  }

  void reset();
  void resetObs();
  void addObservation(State s) { addObservation(s.pos()); };
  void addObservation(Vector2d xy);
  double getScore(bool update);
  void updateScore();
  void resetNearestObs();
  
 private:
  std::priority_queue<double, std::vector<double>, std::greater<double>> obs;
  double currentScore;
  
  Vector2d loc ;
  double val ;
  double obsRadius ;
  double nearestObs ;
  bool observed ;
  size_t curTime ;
  int coupling ;
  
  std::vector<double> nearestObsVector;
  

  int maxConsideredCouple;
 protected:
  void setObserved(bool obs) { observed = obs; }
  void setNearestObs(double nearest) { nearestObs = nearest; }
  
};

#endif // TARGET_H_
