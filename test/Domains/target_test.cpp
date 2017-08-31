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

class TargetTest : public::testing::Test {};

// TEST_F(TargetTest, testDefaults) {
//   Vector2d xy; xy(0) = 1; xy(1) = 2;
//   Target t(xy, 2.5);

  
//   EXPECT_EQ(1,1);
// }

TEST_F(TargetTest, testFirstObservationNoCoupling) {
  Vector2d pos; pos(0) = 0; pos(1) = 0;
  Vector2d xy; xy(0) = 4; xy(1) = 0;

  // Default obs radius is 4.0
  Target t(xy, 3.0);
  t.ObserveTarget(xy);
  
  double expected_nearest = (pos - xy).norm();
  double nearest = t.GetNearestObs();
  
  EXPECT_TRUE(t.IsObserved());
  EXPECT_DOUBLE_EQ(expected_nearest, nearest);
}
// Test_F(TargetTest,) {
//   EXPECT_EQ(1,1);
// }

// TEST_F(TargetTest, testWorseObservationNoCoupling) {

// }

// TEST_F(TargetTest, testBetterObservationNoCoupling) {

// }

// TEST_F(TargetTest, testFirstObservationsCoupling) {}

// TEST_F(TargetTest, testInsufficientObservationsCoupling) {}

// TEST_F(TargetTest, testSufficientObservationsCoupling) {}
// TEST_F(TargetTest, testBetterObservationCoupling) {}
// TEST_F(TargetTest, testWorseObservationsCoupling) {}
