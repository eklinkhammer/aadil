#include "Target.h"

Target::Target(Vector2d xy, double value, int coupling, double obsR,
	       bool obs) : loc(xy), val(value), obsRadius(obsR),
			   nearestObs(DBL_MAX), observed(obs), curTime(-1)  {
  
  nearestObsVector.reserve(coupling);
  
  for (int i = 0; i < coupling; i++) {
    nearestObsVector.push_back(DBL_MAX);
  }
}

void Target::ObserveTarget(Vector2d xy){
  Vector2d diff = xy - loc;
  double d = diff.norm();

  if (d > obsRadius) { return; }
  
  int furthestValidObsIndex = 0;
  double furthestValidObs = DBL_MAX;
  for (size_t i = 0; i < nearestObsVector.size(); i++) {
    if (furthestValidObs < nearestObsVector[i]) {
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
  
  nearestObs = meanObs > nearestObs ? meanObs : nearestObs;
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
  for (size_t i = 0; i < nearestObsVector.size(); i++) {
    nearestObsVector[i] = DBL_MAX;
  }
}
