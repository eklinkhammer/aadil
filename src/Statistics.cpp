/*******************************************************************************
Statistics.h

Calculation of statistics. Intended for use in reportd.


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

#include "Statistics.h"

double mean(std::vector<double> xs) {
  double sum = 0.0;
  double num = (double) xs.size();

  if (num < 1) {
    return 0;
  }
    
  for (const auto& x : xs) {
    sum += x;
  }

  return sum / num;
}

double variance(std::vector<double> xs) {

  double u = mean(xs);
  double n = (double) xs.size();
  double sum = 0.0;
  
  if (n < 1) {
    return 0;
  }

  for (const auto& x : xs) {
    sum += pow(x - u, 2);
  }

  return sum / n;
}

double stddev(std::vector<double> xs) {
  double n = (double) xs.size();
  if (n < 1) {
    return 0;
  }

  return sqrt(variance(xs));
}

double statstderr(std::vector<double> xs) {
  double n = (double) xs.size();
  if (n < 1) {
    return 0;
  }

  return stddev(xs) / sqrt(n);
}
