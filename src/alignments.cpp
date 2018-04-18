/*******************************************************************************
alignments.cpp

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



Alignments::Alignments(std::vector< Objective* > objectives, int numberSamples, double b)
  : objs(objectives), numSamples(numberSamples), biasT(b) {}

void Alignments::addAlignments(Env* env) {
  //std::cout << "Alignments::addAlignments(Env* env) - Entry" << std::endl;
  std::vector<Alignment> scores;
  std::vector<double> values;

  // For each objective, the reward for all samples
  std::vector< std::vector<double> > postVals;

  // For each sample, the change in x position for each agent
  std::vector< std::vector<double> > dx;

  // For each sample, the change in y position for each agent
  std::vector< std::vector<double> > dy;
  
  for (auto& obj : objs) {
    values.push_back(obj->reward(env));
    std::vector<double> postValsForOneObj;
    postVals.push_back(postValsForOneObj);
  }

  //std::cout << "Alignments::addAlignments(Env* env) - Before Sampling" << std::endl;
  for (int i = 0; i < numSamples; i++) {
    std::vector<double> dxs;
    std::vector<double> dys;
    dx.push_back(dxs);
    dy.push_back(dys);
    
    env->reset();
    env->randomStep();

    // Evaluate the sample with all objectives
    for (size_t o = 0; o < objs.size(); o++) {
      postVals[o].push_back(objs[o]->reward(env));
    }

    // Determine the agent direction for each sample
    for (const auto& agent : env->getAgents()) {
      Vector2d initial = agent->getInitialXY();
      Vector2d current = agent->getCurrentXY();

      Vector2d difference = current - initial;

      dx[i].push_back(difference(0));
      dy[i].push_back(difference(1));
    }
  }

  //std::cout << "Alignments::addAlignments(Env* env) - After Sampling" << std::endl;
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

  std::vector<Agent*> agents = env->getAgents();
  for (size_t agent = 0; agent < agents.size(); agent++) {

    std::vector<double> stateV = agents[agent]->getVectorState(env->getCurrentStates());
    Point point{ {stateV[0], stateV[1], stateV[2], stateV[3], stateV[4], stateV[5], stateV[6], stateV[7]} };

    Alignment result;
    for (size_t i = 1; i < objs.size(); i++) {
      std::vector< double > deltaO = deltas[i];    
      for (int j = 0; j < numSamples; j++) {
	double rewardG = deltaG[j];
	double rewardO = deltaO[j];
	result = result.mappend(Alignment(rewardG, rewardO, dx[j][agent], dy[j][agent]));
      }
    }

    tree[point] = result;
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
  double lower = 30;
  double upper = 100;
  
  std::random_device rd;     // only used once to initialise (seed) engine
  std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int> uni(3,10); // guaranteed unbiased
  std::uniform_real_distribution<double> und(lower,upper);
  
  double max_x = und(rng);
  double max_y = und(rng);
  std::vector<double> world = {0, max_x, 0, max_y};
  
  int rovs = uni(rng);
  int pois = uni(rng);

  MultiRover domain(world, 1, 1, pois, "", rovs);

  double r = und(rng);
  if (r > (biasT * (upper - lower) + lower)) {
    domain.setBias(true);
  }
  domain.InitialiseEpoch();
  addAlignments(&domain);
}

Alignment Alignments::getAlignmentsNN(std::vector< double > input) {
  Point point{ {input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]} };
  typename Tree::iterator it = tree.find_nearest_neighbor(point);
  Alignment bestAlign = tree[it->first];
  return bestAlign;
}


