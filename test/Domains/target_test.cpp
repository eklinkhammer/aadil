/*******************************************************************************
target_test.cpp

Unit tests for AADIL common code Domains/Target.cpp class.

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

#include "Domains/Target.h"

#include <iostream>

class TargetTest : public::testing::Test {};

TEST_F(TargetTest, testFirstObservationNoCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;

  // Default obs radius is 4.0
  Target t(pos, 3.0);
  t.ObserveTarget(xy);
  
  double expected_nearest = (pos - xy).norm();
  double nearest = t.GetNearestObs();

  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(expected_nearest, nearest);
}

TEST_F(TargetTest, testFirstObservationTooFarNoCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 2;

  Target t(pos, 3.0, 1, 4, false);
  t.ObserveTarget(xy);
  
  EXPECT_FALSE(t.IsObserved());
}

TEST_F(TargetTest, testWorseObservationNoCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 1, 5, false);
  t.ObserveTarget(xy);

  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4, t.GetNearestObs());

  t.ObserveTarget(xy2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4, t.GetNearestObs());
}

TEST_F(TargetTest, testBetterObservationNoCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 1, 5, false);
  t.ObserveTarget(xy2);

  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4.5, t.GetNearestObs());

  t.ObserveTarget(xy);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4, t.GetNearestObs());
}

TEST_F(TargetTest, testFirstObservationsCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy);
  t.ObserveTarget(xy2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4.25, t.GetNearestObs());
}

TEST_F(TargetTest, testInsufficientObservationsCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy);
  EXPECT_FALSE(t.IsObserved());
}

TEST_F(TargetTest, testBetterObservationCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;
  Vector2d xy3; xy3(0) = 0; xy3(1) = 3;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy);
  t.ObserveTarget(xy2);
  t.ObserveTarget(xy3);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(3.5, t.GetNearestObs());
}

TEST_F(TargetTest, testWorseObservationsCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;
  Vector2d xy3; xy3(0) = 0; xy3(1) = 3;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy3);
  t.ObserveTarget(xy);
  t.ObserveTarget(xy2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(3.5, t.GetNearestObs());
}

TEST_F(TargetTest, testWorseObservationWithTime) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 1, 5, false);
  t.ObserveTarget(xy, 1);

  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4, t.GetNearestObs());

  t.ObserveTarget(xy2, 2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4, t.GetNearestObs());
}

TEST_F(TargetTest, testBetterObservationWithTime) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 1, 5, false);
  t.ObserveTarget(xy2, 1);

  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4.5, t.GetNearestObs());

  t.ObserveTarget(xy, 2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4, t.GetNearestObs());
}

TEST_F(TargetTest, testObserveWithTimeCouplingInsufficentAtTime) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy, 1);
  t.ObserveTarget(xy2, 2);
  EXPECT_FALSE(t.IsObserved());
}

TEST_F(TargetTest, testObserveWithTimeCouplingSufficentAtTime) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy, 2);
  t.ObserveTarget(xy2, 2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4.25, t.GetNearestObs());
}

TEST_F(TargetTest, testObserveWithTimeCouplingSufficentAtTimeThenNewTime) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;
  Vector2d xy2; xy2(0) = 0; xy2(1) = 4.5;
  Vector2d xy3; xy3(0) = 0; xy3(1) = 3;

  Target t(pos, 3.0, 2, 5, false);
  t.ObserveTarget(xy, 1);
  t.ObserveTarget(xy2, 1);
  t.ObserveTarget(xy3, 2);
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(4.25, t.GetNearestObs());
}

TEST_F(TargetTest, testObserveThenReset) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;

  Target t(pos, 3.0);
  t.ObserveTarget(xy, 1);
  t.ResetTarget();

  EXPECT_FALSE(t.IsObserved());
}
