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

vector<State> Env::nextStep(vector< size_t > teamIndex) const {
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

vector<State> Env::step(vector< size_t > teamIndex)  {
  vector<State> nextStates = nextStep(teamIndex);
  applyStep(nextStates);
  return nextStates;
}

void Env::init(vector<State> initStates) {
  historyStates.clear();
  currentStates = initStates;
  historyStates.push_back(currentStates);

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
  init();
  for (auto& target : targets) {
    target.ResetTarget();
  }

  for (auto& agent : agents) {
    if (TeamFormingAgent* t = dynamic_cast<TeamFormingAgent*>(agent)) {
      t->ResetTarget();
    }
  }
}

void Env::init() {
  init(historyStates[0]);
}

void Env::applyNewStateEffects() {
  for (size_t i = 0; i < agents.size(); i++) {
    agents[i]->move(currentStates[i]);
  }

  vector< Vector2d > locs;
  
  for (const auto& state : currentStates) {
    locs.push_back(state.pos());
  }

  for (auto& target : targets) {
    target.observeTargetMultiple(locs);
  }

  for (auto& agent : agents) {
    if (TeamFormingAgent* t = dynamic_cast<TeamFormingAgent*>(agent)) {
      locs.clear();
      Vector2d xy = t->getCurrentXY();
      for (const auto& other : agents) {
	Vector2d otherLoc = other->getCurrentXY();
	if (!(otherLoc(0) == xy(0) && otherLoc(1) == xy(1))) {
	  locs.push_back(otherLoc);
	}
      }

      t->observeTargetMultiple(locs);
    }
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

void Env::randomStep() {
  vector<State> perturbedStates;
  for (const auto& s : currentStates) {
    perturbedStates.push_back(perturbState(s));
  }

  applyStep(perturbedStates);
}

State Env::perturbState(const State s) const {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0,1);

    double dx = dist(e2);
    double dy = dist(e2);

    Vector2d newXY = s.pos();
    newXY(0) += dx;
    newXY(1) += dy;

    State newS(newXY, s.psi());

    return newS;
}
