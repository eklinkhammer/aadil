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


std::vector<Alignment> Alignments::getAlignments(Env* env, size_t agentIndex) {
  //std::cout << "Alignments::addAlignments(Env* env) - Entry" << std::endl;
  std::vector<Alignment> scores;
  std::vector<double> values;

  // For each objective, the reward for all samples
  std::vector< std::vector<double> > postVals;

  // For each sample, the change in x position for each agent
  std::vector< double > dx;

  // For each sample, the change in y position for each agent
  std::vector< double > dy;

  //std::cout << "ADD ALIGNMENTS" << std::endl;
  //std::cout << std::endl;
  //for (auto& obj : objs) {
  //  std::cout << obj->getName() << std::endl;
  //}
  for (auto& obj : objs) {
    //std::cout << "Before obj_reward" << std::endl;
    //std::cout << obj->getName() << std::endl;
    //std::cout << obj->reward(env) << std::endl;
    values.push_back(obj->reward(env));
    //std::cout << "After obj_reward" << std::endl;
    std::vector<double> postValsForOneObj;
    postVals.push_back(postValsForOneObj);
  }

  //std::cout << "Alignments::addAlignments(Env* env) - Before Sampling" << std::endl;
  for (int i = 0; i < numSamples; i++) {
    env->reset();
    env->randomStepSingleAgent();

    // Evaluate the sample with all objectives
    for (size_t o = 0; o < objs.size(); o++) {
      postVals[o].push_back(objs[o]->reward(env));
    }

    Agent* agent = env->getAgents()[agentIndex];
    Vector2d initial = agent->getInitialXY();
    Vector2d current = agent->getCurrentXY();
    Vector2d difference = current - initial;

    dx.push_back(difference(0));
    dy.push_back(difference(1));
  }

  //std::cout << "Alignments::addAlignments(Env* env) - After Sampling" << std::endl;
  std::vector< std::vector< double > > deltas;
  for (size_t i = 0; i < objs.size(); i++) {
    std::vector<double> deltaO;
    for (int j = 0; j < numSamples; j++) {
      double diff = postVals[i][j] - values[i];
      deltaO.push_back(diff);
      // if (i == 0 && diff > 0) {
      // 	std::cout << "Objective gets nonzero reward" << std::endl;
      // 	std::cout << diff << std::endl;
      // }
    }

    deltas.push_back(deltaO);
  }

  std::vector< double > deltaG = deltas[0];

  env->reset();
  Agent* agent = env->getAgents()[agentIndex];
  std::vector<double> stateV = agent->getVectorState(env->getCurrentStates());
  Point point{ {stateV[0], stateV[1], stateV[2], stateV[3], stateV[4], stateV[5], stateV[6], stateV[7]} };

  std::vector<Alignment> results;
  for (size_t i = 1; i < objs.size(); i++) {
    std::vector< double > deltaO = deltas[i];
    Alignment result;
    for (int j = 0; j < numSamples; j++) {
      double rewardG = deltaG[j];
      double rewardO = deltaO[j];
      
      result = result.mappend(Alignment(rewardG, rewardO, dx[j], dy[j], objs[i]->getName()));

      // if (rewardG > 0) {
      // 	std::cout << "G greater than 0." << std::endl;
      // 	std::cout << "Reward G: " << rewardG << " Reward O: " << rewardO << std::endl;
      // 	std::cout << result << std::endl;
      // }
    }
    results.push_back(result);
  }

  return results;
}

void Alignments::addAlignments(Env* env) {
  if (env->getAgents().size() < 1) {
    std::cout << "Environment has no agents. Exiting." << std::endl;
    return;
  }
  
  env->reset();
  Agent* agent = env->getAgents()[0];
  std::vector<double> stateV = agent->getVectorState(env->getCurrentStates());
  Point point{ {stateV[0], stateV[1], stateV[2], stateV[3], stateV[4], stateV[5], stateV[6], stateV[7]} };


  tree[point] = getAlignments(env, 0);
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

std::vector<Alignment> Alignments::getAlignments(MultiRover* domain, size_t agentIndex) {
  //  std::cout << "getAlignments 1" << std::endl;
  Env env(domain->getWorld(), domain->getAgents(), domain->getPOIs(), domain->getNPop());
  // std::cout << "getAlignments 2" << std::endl;
  std::vector<State> states;
  for (const auto& agent : domain->getAgents()) {
    states.push_back(agent->getCurrentState());
  }
  // std::cout << "getAlignments 3" << std::endl;
  env.init(states);
  // std::cout << "getAlignments 4" << std::endl;
  return getAlignments(&env, agentIndex);
}

void Alignments::addAlignments() {
  double lower = 20;
  double upper = 50;
  
  std::random_device rd;     // only used once to initialise (seed) engine
  std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int> uni(5,30); // guaranteed unbiased
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
  std::vector<Alignment> aligns = getAllAlignments(input);
  Alignment mempty;
  return mempty.mconcat(aligns);
}

/**
   KD Tree stores alignment value for each objective. 
 **/
std::vector<Alignment> Alignments::getAllAlignments(std::vector<double> input) {
  Point point{ {input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]} };
  typename Tree::iterator it = tree.find_nearest_neighbor(point);
  std::vector<Alignment> aligns = tree[it->first];
  return aligns;
}

