/*******************************************************************************
Env.cpp

Environment that decouples stepping through the environment, observations,
and awareness of the policies. Comments can be found in header file.

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

#include "Env.h"

Env::Env(vector<double> w, vector<Agent*> team, vector<Target> pois, size_t nPop)
  : world(w), agents(team), targets(pois), teamSize(nPop) {}

Env::~Env() {
  for (size_t i = 0; i < agents.size(); i++) {
    delete agents[i];
    agents[i] = NULL;
  }
}

vector<State> Env::nextStep() const {
  vector<State> jointState;
  for (size_t a = 0; a < agents.size(); a++) {
    State aS = agents[a]->executeNNControlPolicy(teamIndex[a], currentStates);
    jointState.push_back(aS);
  }

  return jointState;
}


void Env::applyStep(vector<State> jointStates) {
  for (size_t i = 0; i < agents.size(); i++) {
    agents[i]->move(jointStates[i]);
  }

  currentStates = jointStates;
  historyStates.push_back(currentStates);
}

vector<State> Env::step()  {
  vector<State> nextStates = nextStep();
  applyStep(nextStates);
  return nextStates;
}

void Env::init(vector<State> initStates, vector<size_t> index) {
  historyStates.clear();
  currentStates = initStates;
  historyStates.push_back(currentStates);

  teamIndex = index;
  
  for (size_t a = 0; a < agents.size(); a++) {
    agents[a]->initialiseNewLearningEpoch(currentStates[a], targets);
  }
}

void Env::reset() {
  init(historyStates[0], teamIndex);
}

double Env::currentReward() const {
  if (rewards.size() == 0) {
    return 0;
  }
  
  return rewards[rewards.size() - 1];
}

double Env::latestStepReward() const {
  double previousReward = rewards.size() > 1 ? rewards[rewards.size() - 2] : 0;
  return currentReward() - previousReward;
}

double Env::estimateRewardOfStep(vector<State> nextStep) const {
  vector< Target > backupTargets = targets;
  vector< Agent* > backupAgents;
  for (const auto& a : agents) {
    backupAgents.push_back(a->copyAgent());
  }

  Env rollOut(world, backupAgents, backupTargets, teamSize);
  rollOut.applyStep(nextStep);

  return rollOut.currentReward();
}
