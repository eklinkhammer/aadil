/*******************************************************************************
TeamFormingAgent.cpp

Rover that only sees Agents. See header file for all documentation.

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

#include "TeamFormingAgent.h"


Vector2d origin(0,0);

TeamFormingAgent::TeamFormingAgent(size_t n, size_t nPop, Fitness f, int tSize)
  : Agent(n, nPop, 4, 12, 2, f), Target(origin, 1, tSize-1), teamSize(tSize) {}

VectorXd TeamFormingAgent::ComputeNNInput(vector<Vector2d> jointState) {
  VectorXd s;
  s.setZero(numIn,1) ;
  MatrixXd Global2Body = RotationMatrix(-currentPsi) ;

  size_t ind = selfIndex(jointState);
  
  Vector2d rovV ;
  rovV.setZero(2,1) ;
  for (size_t i = 0; i < jointState.size(); i++){
    if (i != ind) {
      rovV = jointState[i] - currentXY ;
      Vector2d rovBody = Global2Body*rovV ;
      Vector2d diff = currentXY - rovBody ;
      double d = diff.norm() ;
      double theta = atan2(rovBody(1),rovBody(0)) ;
      size_t q ;
      if (theta >= PI/2.0)
        q = 3 ;
      else if (theta >= 0.0)
        q = 0 ;
      else if (theta >= -PI/2.0)
        q = 1 ;
      else
        q = 2 ;
      s(q) += 1.0/max(d,1.0) ;
    }
  }
  
  return s ;
}

double TeamFormingAgent::getReward() {
  return IsObserved() ? (GetValue() / max(GetNearestObs(), 1.0)) : 0.0;
}

void TeamFormingAgent::DifferenceEvaluationFunction(vector<Vector2d> jointState, double G) {
  // TODO
}

Agent* TeamFormingAgent::copyAgent() const { 
  TeamFormingAgent* copy = new TeamFormingAgent(getNSteps(), getNPop(), getFitness(), teamSize);
  NeuroEvo* nets = GetNEPopulation();
  copy->setNets(nets);
  return copy;
}
