#ifndef ROVER_H_
#define ROVER_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <float.h>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Utilities/Utilities.h"
#include "Domains/Target.h"
#include "Agent.h"

using std::string;
using std::vector;
using std::max;

class Rover : public Agent {
 public:
  Rover(size_t n, size_t nPop, string evalFunc);
  //Rover(size_t n, size_t nPop, Fitness f, size_t nInput, size_t nHidden,
      //size_t nOutput);
  ~Rover() {};

  // Pure Virtual functions
  virtual VectorXd ComputeNNInput(vector<Vector2d> jointState);
  virtual void DifferenceEvaluationFunction(vector<Vector2d>, double);

  // Overriding
  virtual void InitialiseNewLearningEpoch(vector<Target>, Vector2d, double);
    
 private:
  vector<Target> POIs;
};

#endif // ROVER_H_
