/*******************************************************************************
agent_test.cpp

Unit tests for AADIL common code Agents/Agent.cpp virtual class.

TODO: Mock for NeuroEvo or load neural net weights.

Author: Eric Klinkhammer

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

#include "gtest/gtest.h"
#include "Agents/Agent.h"
#include <iostream>

class AgentTest : public::testing::Test {};

class AgentChild : public Agent {
public:
  AgentChild (size_t n, size_t nP, size_t nI, size_t nH, size_t nO, Fitness f)
    : Agent(n, nP, nI, nH, nO, f) {};

  VectorXd ComputeNNInput(vector<Vector2d> jointState) {
    VectorXd s;
    return s;
  };

  void DifferentEvaluationFunction(vector<Vector2d> jointState, double G) {};
};

TEST_F(AgentTest, testInitialiseNewLearningEpoch) {
  
  size_t n = 1, nP = 1, nI = 1, nH = 1, nO = 1;
  AgentChild a(n,nP,nI,nH,nO, Fitness::G);

  Vector2d initXY; initXY(0) = 2; initXY(1) = 5;
  double initPsi = 1.2;

  a.InitialiseNewLearningEpoch(initXY, initPsi);

  EXPECT_EQ(initPsi, a.getInitialPsi());
  EXPECT_EQ(initPsi, a.getCurrentPsi());
  
  EXPECT_EQ(initXY(0), a.getInitialXY()(0));
  EXPECT_EQ(initXY(1), a.getInitialXY()(1));
  EXPECT_EQ(initXY(0), a.getCurrentXY()(0));
  EXPECT_EQ(initXY(1), a.getCurrentXY()(1));
}

TEST_F(AgentTest, testSubstituteCounterfactual) {
  AgentChild a(1,1,1,1,1, Fitness::G);
  Vector2d initXY; initXY(0) = 2; initXY(1) = 5;
  double initPsi = 1.2;

  a.InitialiseNewLearningEpoch(initXY, initPsi);
  
  vector<Vector2d> jointState;
  Vector2d xy1, xy2, xya;
  xy1(0) = 1; xy1(1) = 4; 
  xy2(0) = 3; xy2(1) = 2;
  xya(0) = 2; xya(1) = 6;

  jointState.push_back(xy1);
  jointState.push_back(xy2);
  jointState.push_back(xya);

  vector<Vector2d> newJointState = a.substituteCounterfactual(jointState);

  EXPECT_EQ(initXY(0), newJointState[2](0));
  EXPECT_EQ(initXY(1), newJointState[2](1));
}

