/*******************************************************************************
OnlyPOIRover.cpp

Rover that only sees POIs. See header file for all documentation.

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

#include "OnlyPOIRover.h"

OnlyPOIRover::OnlyPOIRover(size_t n, size_t nPop, Fitness f)
  : Rover(n, nPop, 4, 12, 2, f) {}

VectorXd OnlyPOIRover::ComputeNNInput(vector<Vector2d> jointState) {
  VectorXd s;
  s.setZero(numIn,1) ;
  MatrixXd Global2Body = RotationMatrix(-currentPsi) ;
  
  // Compute POI observation states
  Vector2d POIv ;
  POIv.setZero(2,1) ;

  for (auto& poi: POIs) {
    POIv = poi.GetLocation() - currentXY ;
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
    s(q) += poi.GetValue()/max(d,1.0) ;
  }
  
  return s ;
}
