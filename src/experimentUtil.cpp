/*******************************************************************************
experimentalUtil.h

Experiment Functions

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

#include "experimentUtil.h"

std::vector<double> testDomain(MultiRover* domain, Objective* o, size_t reps) {
  std::vector<double> scores;
  
  std::vector< NeuralNet* > testNets = domain->getBestNNTeam(o);
  std::vector< Agent* > agents = domain->getAgents();
  std::vector< size_t > agentNetIndex(agents.size(),0);
  
  for (size_t agent; agent < agents.size(); agent++) {
    NeuralNet* agentNet = agents[agent]->GetNEPopulation()->GetNNIndex(0);
    agentNet->SetWeights(testNets[agent]->GetWeightsA(),
			 testNets[agent]->GetWeightsB());
  }

  for (size_t r = 0; r < reps; r++) {
    domain->InitialiseEpoch();
    double score = domain->runSim(domain->createSim(domain->getNPop()),
				  agentNetIndex, o);
    scores.push_back(score);
    domain->ResetEpochEvals();
  }

  return scores;
}
void trainDomainEps(MultiRover* domain, YAML::Node exp, Objective* o, size_t eps) {
  int output = intFromYAML(exp, outputS);
  bool toOutput = (output == 1);
  string type = stringFromYAML(exp, typeS);

  int staticOrRandom = intFromYAML(exp, staticOrRandomS);
  bool random = staticOrRandom == 1;
  
  trainDomain(domain, eps, toOutput, 20, type, "", o->getName(), random, o);
}

void trainingCurves(YAML::Node exp, size_t maxEps, size_t trials, size_t tests) {
  MultiRover* domain;
  Objective* o = objFromYAML(exp, objectiveS);

  std::cout << "Training objective " << o->getName() << std::endl;


  for (size_t eps = 0; eps < maxEps + 1; eps += 50) {
    std::vector<double> scores;
    for (size_t t = 0; t < trials; t++) {
      domain = getDomain(exp);
      trainDomainEps(domain, exp, o, eps);
      std::cout << ".";
      std::flush(std::cout);     
      vector<size_t> agents(domain->getNRovers(),0);
      
      for (size_t reps = 0; reps < tests; reps++) {
	domain->InitialiseEpoch();
	domain->ResetEpochEvals();
	scores.push_back(domain->runSim(domain->createSim(domain->getNPop()),
					agents, o));
      }
    }
    std::cout << " Eps: " << eps
	      << " Mean: " << mean(scores)
	      << " stddev: " << stddev(scores)
	      << " stderr: " << statstderr(scores)
	      << std::endl;
  }
  delete domain;
}

// Initialises a MultiRover domain according to the specifications in the node
void setDomain(MultiRover* domain, YAML::Node root) {
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);
  
  domain->setNRovers(nRovs);
  domain->setNPOIs(nPOIs);
  domain->setNSteps(nSteps);
  domain->setCoupling(coupling);
  domain->setType(t);

  domain->initRovers();
  
  int staticOrRandom = intFromYAML(root, staticOrRandomS);
  if (staticOrRandom == 0) {
    YAML::Node targetNode = nodeFromYAML(root, targetsS);
    YAML::Node agentNode = nodeFromYAML(root, agentsS);
    vector<Target> targets = targetsFromYAML(targetNode);
    vector<Vector2d> initXYs = vector2dsFromYAML(agentNode);
  }

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain->setBias(false);
  }
}

MultiRover* getDomain(YAML::Node root) {
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);
  int coupling  = intFromYAML(root, couplingS);

  string type = stringFromYAML(root, typeS);
  AgentType t = stringToAgentType(type);

  size_t cceaPop = size_tFromYAML(root, cceaPopS);

  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};
  
  MultiRover* domain = new MultiRover(world, nSteps, cceaPop, nPOIs, Fitness::G, nRovs, coupling, t);
  
  int staticOrRandom = intFromYAML(root, staticOrRandomS);
  if (staticOrRandom == 0) {
    YAML::Node targetNode = nodeFromYAML(root, targetsS);
    YAML::Node agentNode = nodeFromYAML(root, agentsS);
    vector<Target> targets = targetsFromYAML(targetNode);
    vector<Vector2d> initXYs = vector2dsFromYAML(agentNode);
    domain->InitialiseEpochFromVectors(targets, initXYs);
  }

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domain->setBias(false);
  }

  return domain;
}

vector<NeuroEvo*> readNetworksFromFile(const std::string filename, size_t numIn,
				       size_t numH, size_t numOut, size_t numA,
				       size_t numPop) {
  
  vector<NeuralNet*> loadedNN = NeuralNet::loadNNFromFile(filename, numIn,
							  numH, numOut);

  vector<NeuroEvo*> agentsNNs;
  for (size_t i = 0; i < numA; i++) {

    NeuroEvo* agentNNs = new NeuroEvo(numIn, numOut, numH, numPop);
    for (size_t j = 0; j < numPop; j++) {
      agentNNs->GetNNIndex(j)->SetWeights(loadedNN[i]->GetWeightsA(),
					  loadedNN[i]->GetWeightsB());
    }
    agentsNNs.push_back(agentNNs);
  }

  for (size_t i = 0; i < loadedNN.size(); i++) {
    delete loadedNN[i];
    loadedNN[i] = NULL;
  }

  return agentsNNs;
}

inline bool exists_test3 (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// Given an experiment node, finds the highest performing neural network
//   team to satisfy that objective.
vector< NeuralNet* > bestTeamForObjective(YAML::Node expNode, std::string expKey,
					  MultiRover* domain, Objective* o,
					  std::string fileDir) {
  vector< NeuralNet* > bestTeam;

  domain = getDomain(expNode);
  
  std::cout << "Getting the best team for " << expKey << std::endl;

  bool train = boolFromYAML(expNode, trainS);
  std::string filename = stringFromYAML(expNode, netfileS);
  if (!train && exists_test3(filename)) {
    //std::cout << "Reading in training policies from " << filename << std::endl;
    Agent* agent = domain->getAgents()[0];
    size_t numIn = agent->getNumIn();
    size_t numHidden = agent->getNumHidden();
    size_t numOut = agent->getNumOut();
    domain->loadNNs(filename, numIn, numHidden, numOut);
    domain->InitialiseEpoch();
    bestTeam = domain->getBestNNTeam(o);
  } else {
    vector<double> scores;
    double bestScore = 0.0;
    std::cout << "Training neural network team for " << expKey;
    std::flush(std::cout);

    // Trial consists of training domain, getting the best neural network team, and
    //  then testing that network team on a variety of different worlds.
    // The performance of the best team across the different trials is saved
    for (size_t trial = 0; trial < 10; trial++) {
      domain = getDomain(expNode);
      trainDomain(domain, expNode, expKey, fileDir, o);
      vector< NeuralNet* > trialNets = domain->getBestNNTeam(o);
      vector<size_t> agents(domain->getNRovers(), 0);
      vector< Agent* > as = domain->getAgents();
      
      for (size_t a = 0; a < as.size(); a++) {
	NeuralNet* agentZ = as[a]->GetNEPopulation()->GetNNIndex(0);
	agentZ->SetWeights(trialNets[a]->GetWeightsA(),
			   trialNets[a]->GetWeightsB());
      }
      
      for (size_t reps = 0; reps < 50; reps++) {
      	domain->InitialiseEpoch();
      	domain->ResetEpochEvals();
      	scores.push_back(domain->runSim(domain->createSim(domain->getNPop()), agents, o));
      }
      
      double newScore = mean(scores);
      if (newScore >= bestScore) {
	bestScore = newScore;
	bestTeam = trialNets;

      }
      std::cout << ".";
      std::flush(std::cout);
    }
    if (exists_test3(filename)) {
      std::string command = "rm " + filename;
      system(command.c_str());
    }
    domain->OutputControlPolicies(filename);
    std::cout << " Done. Final best score: " << bestScore << std::endl;
  }
  return bestTeam;
}

AgentType stringToAgentType(std::string type) {
  if      (type == "A") return AgentType::A;
  else if (type == "P") return AgentType::P;
  else if (type == "R") return AgentType::R;
  else if (type == "M") return AgentType::M;
  else if (type == "E") return AgentType::E;
  else if (type == "C") return AgentType::C;
  else                  return AgentType::R;
}

void trainDomain(MultiRover* domain, size_t epochs, bool output, int outputPeriod,
		 string exp, string topDir, string id, bool init, Objective* o) {
  for (size_t n = 0; n < epochs; n++) {
    if (output && (n % outputPeriod == 0 || n == epochs - 1)) {
      std::cout << "Training " << id << " Episode " << n << "...";
      domain->setVerbose(true);
    } else {
      domain->setVerbose(false);
    }
    if (n == epochs - 1) {
      configureOutput(domain, topDir, id);
    }

    trainDomainOnce(domain, (n==0), init, o);
  }
}

void trainDomain(MultiRover* domain, YAML::Node root, std::string key, std::string topDir, Objective* o) {
  size_t nEps = size_tFromYAML(root, nEpsS);
  int output = intFromYAML(root, outputS);
  bool toOutput = (output == 1);
  
  string type = stringFromYAML(root, typeS);
  // AgentType t = stringToAgentType(type);

  int staticOrRandom = intFromYAML(root, staticOrRandomS);
  bool random = staticOrRandom == 1;

  trainDomain(domain, nEps, toOutput, 20, type, topDir, key, random, o);
}

void trainDomainOnce(MultiRover* domain, bool evolve, bool init, Objective* o) {
  domain->EvolvePolicies(evolve);
  if (init) {
    domain->InitialiseEpoch();
  }
  domain->ResetEpochEvals();
  domain->SimulateEpoch(true, o);
}

std::vector<NeuralNet> getTeam(MultiRover* domain) {
  std::vector<NeuralNet*> myNets = domain->getNNTeam();
  std::vector<NeuralNet> returnNets;
  
  for (auto& net : myNets) {
    NeuralNet newNet(net->getNI(), net->getNO(), net->getNH());
    newNet.SetWeights(net->GetWeightsA(), net->GetWeightsB());
    returnNets.push_back(newNet);
  }
  return returnNets;
}

void configureOutput(MultiRover* domain, string fileDir, string id) {
  string resultFile = fileDir + "/" + id + "_results";
  string trajFile   = fileDir + "/" + id + "_trajectory";
  string poiFile    = fileDir + "/" + id + "_POIs";
  string choiceFile = fileDir + "/" + id + "_choices";
  domain->OutputPerformance(resultFile);
  domain->OutputTrajectories(trajFile, poiFile, choiceFile);
}

void inspectAgent(MultiRover* domain, VectorXd input) {
  vector<Agent*> agents = domain->getAgents();

  //for (const auto& agent : agents) {
    vector<State> states = agents[0]->allNextGetNextState(input);
    std::cout << "Current State" << std::endl;
    std::cout << agents[0]->getCurrentState() << std::endl;
    //for (const auto& state : states) {
      std::cout << states[0] << std::endl;
      //}
    //}
}

/**
   Each agent uses the cluster sample method to select alignment, and then
     uses said alignment's action.
 **/
std::vector<double> alignmentGuidedAction(MultiRover* domain, Alignments* as,
					  std::vector<std::vector<size_t>> inds,
					  std::vector<std::vector<NeuralNet*>>
					  teams, size_t trials, Objective* r,
					  std::vector<size_t> agentSelect) {
  std::vector<double> results;
  vector< Agent* > roverTeam;
  
  for (size_t i = 0; i < domain->getNRovers(); i++) {
    vector<NeuralNet*> netsAgent;
    for (size_t j = 0; j < teams.size(); j++) {
      int netI = teams[j].size() > i ? i : 0;
      netsAgent.push_back(teams[j][netI]);
    }
    roverTeam.push_back(new AlignmentLearningAgent(netsAgent, as, inds, domain->getNPop(), 1.0, 1.0));
  }
  domain->setTeam(roverTeam);

  for (size_t rep = 0; rep < trials; rep++) {
    domain->InitialiseEpoch();
    results.push_back(domain->runSim(domain->createSim(domain->getNPop()),
				     agentSelect, r));
    //std::cout << results[rep] << std::endl;
    domain->ResetEpochEvals();
  }

  return results;
}

/**
   Each agent uses the cluster sample method to select alignment, and then
     uses said alignment's action. The agent then learns over time.
 **/
std::vector<double> alignmentLearning(MultiRover* domain, Alignments* as,
				      std::vector<std::vector<size_t>> inds,
				      std::vector<std::vector<NeuralNet*>>
				      teams, size_t trials, Objective* r,
				      std::vector<size_t> agentSelect) {
  std::vector<double> results;
  vector< Agent* > roverTeam;
  
  for (size_t i = 0; i < domain->getNRovers(); i++) {
    vector<NeuralNet*> netsAgent;
    for (size_t j = 0; j < teams.size(); j++) {
      int netI = teams[j].size() > i ? i : 0;
      netsAgent.push_back(teams[j][netI]);
    }
    roverTeam.push_back(new AlignmentLearningAgent(netsAgent, as, inds, domain->getNPop(), 1.0, 1.0));
  }
  domain->setTeam(roverTeam);

  trainDomain(domain, 1000, false, 20, "ABC", "", "ABC", true, r);
  results = testDomain(domain, r, trials);

  return results;
}

/**
   Each agent uses the cluster sample method to select alignment, and then
     uses said alignment to select a policy.
 **/
std::vector<double> alignmentGuidedPolicy(MultiRover* domain, Alignments* as,
					  std::vector<std::vector<size_t>> inds,
					  std::vector<std::vector<NeuralNet*>>
					  allNets, size_t numObj,
					  size_t trials, Objective* r,
					  std::vector<size_t> agentSelect) {

  std::vector<double> results;
  std::vector<Agent*> roverTeam;
  //std::cout << "Debug 1" << std::endl;
  for (size_t i = 0; i < domain->getNRovers(); i++) {
    vector<NeuralNet*> nets;
    //std::cout << "Debug 2" << std::endl;
    for (size_t n = 0; n < numObj; n++) {
      size_t netI = allNets[n].size() > i ? i : 0;
      nets.push_back(allNets[n][netI]);
    }
    roverTeam.push_back(new AlignmentGuidedAgent(nets, as, inds));
  }

  //std::cout << "Debug 1" << std::endl;
  domain->setTeam(roverTeam);
  //std::cout << "Debug 1" << std::endl;
  for (size_t rep = 0; rep < trials; rep++) {
    domain->InitialiseEpoch();
    results.push_back(domain->runSim(domain->createSim(domain->getNPop()),
				     agentSelect, r));
    //std::cout << results[rep] << std::endl;
    domain->ResetEpochEvals();
  }
  //  results = testDomain(domain, r, trials);
  //std::cout << "Debug 1" << std::endl;
  return results;
}

/**
 directAlignmentUse

 Each agent explicitly calculates alignment.
 **/
std::vector<double> directAlignmentUse(MultiRover* domain, YAML::Node expNode,
				       Objective* r,
				       size_t samples, double param,
				       std::vector<Objective*> objs, size_t reps) {
  Alignments alignments(objs, samples, param);

  std::vector<double> results;

  for (size_t trial = 0; trial < reps; trial++) {
    domain = getDomain(expNode);
    domain->InitialiseEpoch();
    domain->init();

    std::vector< std::vector< State >> history;
    history.push_back(domain->getInitialStates());
    
    std::vector<Agent*> agentTeam = domain->getAgents();

    //std::cout << "directAlignmentUse";
    //std::flush(std::cout);
    for (size_t step = 0; step < domain->getNSteps(); step++) {
      for (size_t agentI = 0; agentI < domain->getNRovers(); agentI++) {
	Alignment chosenAlignment;
	std::vector<Alignment> alignVec = alignments.getAlignments(domain, agentI);
	chosenAlignment = chosenAlignment.mconcat(alignVec);

	Vector2d currentPos = agentTeam[agentI]->getCurrentState().pos();
	Vector2d delta(chosenAlignment.getVecX(), chosenAlignment.getVecY());

	State s(currentPos + delta, 0);
	agentTeam[agentI]->move(s);
      }

      history.push_back(domain->getCurrentJointState());
      //std::cout << *domain << std::endl;
      //std::cout << domain->reward(r) << std::endl;
      //std::cout << domain->reward(r, history) << std::endl;
    }

    results.push_back(domain->reward(r, history));
  }

  return results;
}

