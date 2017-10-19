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
  : Env(w, team, pois, nPop, "Default") {}

Env::Env(vector<double> w, vector<Agent*> team, vector<Target> pois, size_t nPop, string name)
  : world(w), agents(team), targets(pois), teamSize(nPop), idstring(name) {}

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
  applyNewStateEffects();
  curTime += 1;
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

  while(agents.size() < initStates.size()) {
    agents.push_back(agents[0]->copyAgent());
  }

  while(agents.size() > initStates.size()) {
    agents.pop_back();
  }
  
  for (size_t a = 0; a < initStates.size(); a++) {
    agents[a]->initialiseNewLearningEpoch(currentStates[a], targets);
  }

  curTime = 0;
}

void Env::reset() {
  init(historyStates[0], teamIndex);

  for (auto& target : targets) {
    target.ResetTarget();
  }

  for (auto& agent : agents) {
    if (TeamFormingAgent* t = dynamic_cast<TeamFormingAgent*>(agent)) {
      t->ResetTarget();
    }
  }
}

void Env::init(vector< size_t > index) {
  init(historyStates[0], index);
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

double Env::estimateRewardOfStep(vector<State> nextStep) {
  vector< Target > backupTargets = targets;
  vector< Agent* > backupAgents;
  for (const auto& a : agents) {
    Agent* copy = a->copyAgent();
    backupAgents.push_back(copy);
  }
  
  Env rollOut(world, backupAgents, backupTargets, teamSize);
  rollOut.init(getCurrentStates(), teamIndex);
  rollOut.applyStep(nextStep);

  return rollOut.currentReward();
}

// For reasons I don't understand, this does not work.
double Env::estimateRewardOfNextStep() {
  return estimateRewardOfStep(nextStep());
}

void Env::applyNewStateEffects() {
  for (size_t i = 0; i < agents.size(); i++) {
    agents[i]->move(currentStates[i]);
  }
  for (const auto& state : currentStates) {
    applyStateEffect(state);
  }

  double reward = calculateG();
  rewards.push_back(reward);
} 

double Env::calculateG() const {
  double G = 0.0;

  for (auto& a : agents) {
    G += a->getReward();
  }

  if (TeamFormingAgent* t = dynamic_cast<TeamFormingAgent*>(agents[0])) {
    return G;
  }
  
  for (auto& t : targets) {
    G +=  t.IsObserved() ? (t.GetValue() / max(t.GetNearestObs(), 1.0)) : 0.0;
  }

  return G;
}

void Env::applyStateEffect(State s) {
  Vector2d xy = s.pos();

  for (auto& agent : agents) {
    if (TeamFormingAgent* t = dynamic_cast<TeamFormingAgent*>(agent)) {
      Vector2d otherLoc = t->getCurrentXY();
      if (!(otherLoc(0) == xy(0) && otherLoc(1) == xy(1))) {
	t->ObserveTarget(s.pos(), curTime);
      }
    }
  }

  for (auto& target : targets) {
    target.ObserveTarget(xy, curTime);
  }
}

void Env::setTargetLocations(vector< Target > locs) {
  // If this environment has no targets, add none
  if (targets.size() == 0) return;

  // If this env has fewer targets, add copies (this preserves coupling, and the
  //   reward function for the env).
  while (targets.size() < locs.size()) {
    targets.push_back(targets[0]);
  }

  // If the env has more targets, delete them
  while (locs.size() < targets.size()) {
    targets.pop_back();
  }

  // Set the targets to have the same locations. All other (reward determining) features
  //   are still the same
  for (size_t i = 0; i < locs.size(); i++) {
    targets[0].setLocation(locs[i].GetLocation());
  }
}
