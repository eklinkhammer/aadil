/*******************************************************************************
alignment_test.cpp

Unit tests for AADIL common code alignment.h

I am still undecided on how alignments should actually be calculated. These
  tests test the mappend function.

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
#include "alignment.h"
#include <random>

class AlignmentTest : public::testing::Test {};

// mappend mempty x = x = mappend x mempty
TEST_F(AlignmentTest, testMappendIdentity) {
  Alignment mempty;
  for (int trial = 0; trial < 100; trial++) {
    Alignment x(rand(), rand());
    Alignment r = x.mappend(mempty);
    EXPECT_DOUBLE_EQ(x.alignScore(), r.alignScore());
    EXPECT_DOUBLE_EQ(x.alignMag(), r.alignMag());

    r = mempty.mappend(x);
    EXPECT_EQ(x.alignScore(), r.alignScore());
    EXPECT_DOUBLE_EQ(x.alignMag(), r.alignMag());
    
  }
}

// mappend x y = mappend y x
TEST_F(AlignmentTest, testMappendAssociative) {
  for (int trial = 0; trial < 100; trial++) {
    Alignment x(rand(), rand());
    Alignment y(rand(), rand());

    Alignment zl = x.mappend(y);
    Alignment zr = y.mappend(x);
    EXPECT_EQ(zl.alignScore(), zr.alignScore());
    EXPECT_DOUBLE_EQ(zl.alignMag(), zr.alignMag());
  }
}

TEST_F(AlignmentTest, testMappend) {
  for (int trial = 0; trial < 500; trial++) {
    Alignment x(rand(), rand());
    Alignment y(rand(), rand());

    Alignment z = x.mappend(y);
    Alignment c;
    if (x.alignScore() > y.alignScore()) {
      c = y;
    } else if (x.alignScore() < y.alignScore()) {
      c = x;
    } else {
      c = (x.alignMag() > y.alignMag()) ? x : y;
    }
      
    EXPECT_EQ(z.alignScore(), c.alignScore());
    EXPECT_DOUBLE_EQ(z.alignMag(), c.alignMag());
  }
}
