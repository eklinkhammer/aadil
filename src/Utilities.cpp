#include "Utilities/Utilities.h"

namespace easymath{
double rand_interval(double low, double high){
  double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  return r*(high - low) + low;
}

double sum(std::vector<double> v){
  double t = 0 ;
  for (size_t i = 0; i < v.size(); i++)
    t += v[i] ;
  return t ;
}
} // namespace easymath
