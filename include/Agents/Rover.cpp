#include "Rover.h"

Rover::Rover(size_t n, size_t nPop, Fitness f) : Rover(n, nPop, 8, 16, 2, f) {}

Rover::Rover(size_t n, size_t nPop, size_t nInput, size_t nHidden, size_t
	     nOut, Fitness f) : Agent(n, nPop, nInput, nHidden, nOut, f) {}

// Initial simulation parameters, includes setting initial rover position, POI
//   positions and values, and clearing the evaluation storage vector
void Rover::InitialiseNewLearningEpoch(vector<Target> pois, Vector2d xy,
				       double psi) {

  Agent::InitialiseNewLearningEpoch(xy, psi);
  
  POIs.clear();
  for (size_t i = 0; i < pois.size(); i++){
    POIs.push_back(pois[i]) ;
  }
}

// Compute the NN input state given the rover locations and the POI locations and values in the world
VectorXd Rover::ComputeNNInput(vector<Vector2d> jointState) {
  VectorXd s;
  s.setZero(numIn,1) ;
  MatrixXd Global2Body = RotationMatrix(-currentPsi) ;
  
  // Compute POI observation states
  Vector2d POIv ;
  POIv.setZero(2,1) ;
  for (size_t i = 0; i < POIs.size(); i++){
    POIv = POIs[i].GetLocation() - currentXY ;
    Vector2d POIbody = Global2Body*POIv ;
    Vector2d diff = currentXY - POIbody ;
    double d = diff.norm() ;
    double theta = atan2(POIbody(1),POIbody(0)) ;
    size_t q ;
    if (theta >= PI/2.0)
      q = 3 ;
    else if (theta >= 0.0)
      q = 0 ;
    else if (theta >= -PI/2.0)
      q = 1 ;
    else
      q = 2 ;
    s(q) += POIs[i].GetValue()/max(d,1.0) ;
  }

  // Compute rover observation states
  size_t ind = selfIndex(jointState);
  
  Vector2d rovV ;
  rovV.setZero(2,1) ;
  for (size_t i = 0; i < jointState.size(); i++){
    if (i != ind){
      rovV = jointState[i] - currentXY ;
      Vector2d rovBody = Global2Body*rovV ;
      Vector2d diff = currentXY - rovBody ;
      double d = diff.norm() ;
      double theta = atan2(rovBody(1),rovBody(0)) ;
      size_t q ;
      if (theta >= PI/2.0)
        q = 7 ;
      else if (theta >= 0.0)
        q = 4 ;
      else if (theta >= -PI/2.0)
        q = 5 ;
      else
        q = 6 ;
      s(q) += 1.0/max(d,1.0) ;
    }
  }
  
  return s ;
}

void Rover::DifferenceEvaluationFunction(vector<Vector2d> jointState, double G){
  double G_hat = 0 ;
  jointState = substituteCounterfactual(jointState);
  
  for (size_t i = 0; i < jointState.size(); i++)
    for (size_t j = 0; j < POIs.size(); j++)
      POIs[j].ObserveTarget(jointState[i]) ;
       
  for (size_t j = 0; j < POIs.size(); j++){
    G_hat += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
    POIs[j].ResetTarget() ;
  }
  
  stepwiseD += (G-G_hat);
}

Agent* Rover::copyAgent() const {
  return new Rover(*this);
}
