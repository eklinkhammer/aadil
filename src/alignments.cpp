/*******************************************************************************
alignments.h

Given objectives, computes alignment values between them.

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

#include "alignments.h"

Alignments::Alignments(std::vector< Objective* > objectives, int numberSamples)
  : objs(objectives), numSamples(numberSamples) {}

void Alignments::addAlignments(Env* env) {
  std::vector<Alignment> scores;
  std::vector<double> values;
  std::vector< std::vector<double> > postVals;

  for (auto& obj : objs) {
    values.push_back(obj->reward(env));
    std::vector<double> postValsForOneObj;
    postVals.push_back(postValsForOneObj);
  }

  for (int i = 0; i < numSamples; i++) {
    env->reset();
    env->randomStep();

    for (size_t o = 0; o < objs.size(); o++) {
      postVals[o].push_back(objs[o]->reward(env));
    }
  }

  std::vector< std::vector< double > > deltas;
  for (size_t i = 0; i < objs.size(); i++) {
    std::vector<double> deltaO;
    for (int j = 0; j < numSamples; j++) {
      double diff = postVals[i][j] - values[i];
      deltaO.push_back(diff);
    }

    deltas.push_back(deltaO);
  }

  std::vector< double > deltaG = deltas[0];
  for (size_t i = 1; i < objs.size(); i++) {
    std::vector< double > deltaO = deltas[i];
    
    Alignment result;
    for (int j = 0; j < numSamples; j++) {
      double rewardG = deltaG[j];
      double rewardO = deltaO[j];
      result = result.mappend(Alignment(rewardG, rewardO));
    }
    scores.push_back(result);
  }

  for (const auto& agent : env->getAgents()) {
    std::vector<double> stateV = agent->getVectorState(env->getCurrentStates());
    std::pair<std::vector<double>, std::vector<Alignment>> keyVal(stateV, scores);
    alignments.insert(keyVal);
  }
}

void Alignments::addAlignments(int num) {
  for (int i = 0; i < num; i++) {
    addAlignments();
  }
}

void Alignments::addAlignments(MultiRover* domain) {
  Env env(domain->getWorld(), domain->getAgents(), domain->getPOIs(), domain->getNPop());
  env.init(domain->getInitialStates());
  addAlignments(&env);
}

void Alignments::addAlignments() {
  std::random_device rd;     // only used once to initialise (seed) engine
  std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int> uni(3,10); // guaranteed unbiased
  std::uniform_real_distribution<double> und(10,100);
  
  double max_x = und(rng);
  double max_y = und(rng);
  std::vector<double> world = {0, max_x, 0, max_y};
  
  int rovs = uni(rng);
  int pois = uni(rng);

  MultiRover domain(world, 1, 1, pois, "", rovs);
  domain.InitialiseEpoch();
  addAlignments(&domain);
}

// void Alignments::addAlignments() {
//   std::random_device rd;     // only used once to initialise (seed) engine
//   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
//   std::uniform_int_distribution<int> uni(3,10); // guaranteed unbiased
//   std::uniform_real_distribution<double> und(10,100);
  
//   double max_x = und(rng);
//   double max_y = und(rng);
//   std::vector<double> world = {0, max_x, 0, max_y};
  
//   int rovs = uni(rng);
//   int pois = uni(rng);

//   std::vector<Agent*> agents;
//   for (size_t i = 0; i < rovs; i++) {
//     agents.push_back(new Rover(1,1,Fitness::G));
//   }

//   double rangeX = world[1] - world[0] ;
//   double rangeY = world[3] - world[2] ;

//   std::vector<State> initialStates;
//   for (size_t i = 0; i < rovs; i++){
//     Vector2d initialXY ;
//     initialXY(0) = rand_interval(world[0]+rangeX/3.0,world[1]-rangeX/3.0);
//     initialXY(1) = rand_interval(world[2]+rangeY/3.0,world[3]-rangeX/3.0);
//     double initialPsi = rand_interval(-PI,PI) ;
//     State s(initialXY, initialPsi);
//     initialStates.push_back(s);
//   }
  
//   std::vector<Target> POIs;
//   for (size_t p = 0; p < pois; p++) {
//     Vector2d xy ;
//     double x, y ;
//     bool accept = false ;
//     while (!accept){
//       x = rand_interval(world[0],world[1]) ;
//       y = rand_interval(world[2],world[3]) ;
      
//       accept = !(x > world[0]+rangeX/3.0 &&
// 		 x < world[1]-rangeX/3.0 &&
// 		 y > world[2]+rangeY/3.0 &&
// 		 y < world[3]-rangeX/3.0);
//     }
    
//     xy(0) = x ; // x location
//     xy(1) = y ; // y location
//     double v = rand_interval(1,10) ; // value
//     POIs.push_back(Target(xy, 1, 1));
//   }

//   Env env(world, agents, POIs, 1);
//   env.init(initialStates);
//   addAlignments(&env);
// }


// This linearly searches. Should be using a K-d tree. Convert at later date
std::vector< Alignment > Alignments::getAlignmentsNN(std::vector< double > inputState) {
  double minDistance = DBL_MAX;
  std::vector< Alignment > bestAlign;
  
  for (auto it = alignments.begin(); it != alignments.end(); ++it) {
    double dist = distance(inputState, it->first);
    if (dist < minDistance) {
      bestAlign = it->second;
      minDistance = dist;
    }
  }

  return bestAlign;
}

double Alignments::distance(std::vector<double> vec1, std::vector<double> vec2) {
  if (vec1.size() != vec2.size()) return DBL_MAX;

  double squared_dist = 0;
  for (size_t i = 0; i < vec1.size(); i++) {
    double diff = vec2[i] - vec1[i];
    squared_dist += diff*diff;
  }

  return squared_dist;
}


