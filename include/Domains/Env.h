/*******************************************************************************
Env.h

Environment that decouples stepping through the environment, observations,
and awareness of the policies.

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

#ifndef ENV_H_
#define ENV_H_

#include <vector>
#include <Eigen/Eigen>

#include "Agents/Agent.h"
#include "State.h"
#include "Agents/TeamFormingAgent.h"

using std::vector;
using std::string;

class Env {
 public:
  Env(vector<double>, vector<Agent*>, vector<Target>, size_t);

  Env(vector<double>, vector<Agent*>, vector<Target>, size_t, string);
  
  // Returns the result of stepping the simulation forward one step. Does not
  //   actually step the simulation forward.
  vector<State> nextStep(vector< size_t >) const;

  // Steps the simulation forward into the passed jointState (note,
  // the input is NOT a delta)
  void applyStep(vector<State>);

  // Steps the simulation forward one step, and returns the new jointState.
  // Each agent will use neural net specified by index vector.
  vector<State> step(vector<size_t>);
  
  // Resets all agents to starting location, clears history
  void reset();

  // Resets, to a specified set of starting locations and angles.

  void init(vector<State>);

  // Resets, preserves starting locations.
  void init();
  
  string getID() const { return idstring; }
  void setID(string label) { idstring = label; }
  
  vector< Agent* > getAgents()    const { return agents; }
  vector< double > getWorld()     const { return world; }
  vector< Target > getTargets()   const { return targets; };

  vector< State >         getCurrentStates() const { return currentStates; }
  vector< vector<State> > getHistoryStates() const { return historyStates; }

  void setTargetLocations(vector< Target > locs);

  void randomStep();
 private:
  size_t teamSize;
  vector< double > world;
  vector< Agent* > agents;
  vector< Target > targets;
  vector<State> currentStates;
  vector< vector<State> > historyStates;

  size_t curTime;
  string idstring;

  void applyNewStateEffects();
  State perturbState(const State) const;
};

#endif // ENV_H_
