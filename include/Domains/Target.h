#ifndef TARGET_H_
#define TARGET_H_

#include <iostream>
#include <float.h>
#include <Eigen/Eigen>
#include <vector>

using namespace Eigen;

class Target{
 public:

  Target(Vector2d xy, double v) : Target(xy, v, 1) {}
  Target(Vector2d xy, double v, int c) : Target(xy, v, c, 4.0, false) {}  

  Target(Vector2d xy, double value, int coupling, double obsR, bool obs);  
  ~Target() {}

  
  Vector2d GetLocation() { return loc; }
  double GetValue() { return val; }
  double GetNearestObs() { return nearestObs; }
  bool IsObserved() { return observed; }

  void ObserveTarget(Vector2d xy);
  void ObserveTarget(Vector2d xy, size_t t);
  void ResetTarget();

 private:
    Vector2d loc ;
    double val ;
    double obsRadius ;
    double nearestObs ;
    bool observed ;
    size_t curTime ;
    int coupling ;
    
    std::vector<double> nearestObsVector;

    void resetNearestObs();
};

#endif // TARGET_H_
