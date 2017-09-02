#include "Rover.h"

Rover::Rover(size_t n, size_t nPop, string evalFunc): nSteps(n), popSize(nPop){
  numIn = 8 ; // hard coded for 4 element input (body frame quadrant decomposition)
  numOut = 2 ; // hard coded for 2 element output [dx,dy]
  numHidden = 16 ;
  RoverNE = new NeuroEvo(numIn, numOut, numHidden, nPop) ;
  
  if (evalFunc.compare("D") == 0)
    isD = true ;
  else if (evalFunc.compare("G") == 0)
    isD = false ;
  else{
    std::cout << "ERROR: Unknown evaluation function type [" << evalFunc << "], setting to global evaluation!\n" ;
    isD = false ;
  }
  windowSize = nSteps/10 ; // hardcoded running average window size to be 1/10 of full experimental run
  rThreshold.push_back(0.01) ;
  rThreshold.push_back(0.3) ; // hardcoded reward threshold, D logs for 1000 step executions suggest this is a good value
  pomdpAction = 0 ; // initial action is always to not ask for help
  stateObsUpdate = false ; // true if human assistance has redefined NN control policy state calculation
}

// Initial simulation parameters, includes setting initial rover position, POI
//   positions and values, and clearing the evaluation storage vector
void Rover::InitialiseNewLearningEpoch(vector<Target> pois, Vector2d xy,
				       double psi) {

  InitialiseNewLearningEpoch(xy, psi);
  POIs.clear();
  
  for (size_t i = 0; i < pois.size(); i++){
    POIs.push_back(pois[i]) ;
  }
}

// Compute the NN input state given the rover locations and the POI locations and values in the world
VectorXd Rover::ComputeNNInput(vector<Vector2d> jointState){
  VectorXd s ;
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
//    std::cout << "Rover global state: (" << xy[0] << "," << xy[1] << "," << psi*180.0/PI 
//    << "), POI global location: (" << POIs[i].GetLocation()[0] << "," << POIs[i].GetLocation()[1] 
//    << "), POI body location: (" << POIbody[0] << "," << POIbody[1]
//    << "), bearing: " << theta*180.0/PI << ", quadrant: " << q << std::endl ;
  }
//  std::cout << "State: [" << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "]\n" ;

  // Compute rover observation states
  size_t ind = 0 ; // stores agent's index in the joint state
  double minDiff = DBL_MAX ;
  for (size_t i = 0; i < jointState.size(); i++){
    double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
    if (diff < minDiff){
      minDiff = diff ;
      ind = i ;
    }
  }
  
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
  size_t ind = 0 ; // stores agent's index in the joint state
  double minDiff = DBL_MAX ;
  for (size_t i = 0; i < jointState.size(); i++){
    double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
    if (diff < minDiff){
      minDiff = diff ;
      ind = i ;
    }
  }
  
  // Replace agent state with counterfactual
  jointState[ind](0) = initialXY(0) ;
  jointState[ind](1) = initialXY(1) ;
  for (size_t i = 0; i < jointState.size(); i++)
    for (size_t j = 0; j < POIs.size(); j++)
      POIs[j].ObserveTarget(jointState[i]) ;
       
  for (size_t j = 0; j < POIs.size(); j++){
    G_hat += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
    POIs[j].ResetTarget() ;
  }
  
  stepwiseD += (G-G_hat) ;
  if (runningAvgR.size() == windowSize)
    runningAvgR.pop_front() ;
  runningAvgR.push_back(G) ;
}
